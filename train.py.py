# ============================================================
# MedSAM Active Prompt Learning — WINDOWS NATIVE VERSION
# Reads directly from C:\Abdominal CT scans\
# Writes outputs to C:\Abdomen CT code\outputs\
# ============================================================

# --- Dependency check (run this first, or use pip manually) ---
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip install transformers scikit-image scikit-learn pandas matplotlib scipy seaborn psutil

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob, random, gc, json, shutil, zipfile
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from scipy.stats import ttest_rel
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import SamModel
import psutil

# ============================================================
# PATHS — Windows-native
# ============================================================
IMAGE_DIR = r"C:\Abdominal CT scans\abdominal\abdominal\images"
LABEL_DIR = r"C:\Abdominal CT scans\abdominal\abdominal\labels"
OUTPUT_DIR = r"C:\Abdomen CT code\outputs"
CKPT_ROOT = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"IMAGE_DIR  = {IMAGE_DIR}")
print(f"LABEL_DIR  = {LABEL_DIR}")
print(f"OUTPUT_DIR = {OUTPUT_DIR}\n")

# Verify paths exist
if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
if not os.path.isdir(LABEL_DIR):
    raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

# ============================================================
# 0. PRE-FLIGHT RAM CHECK
# ============================================================
ram = psutil.virtual_memory()
print(f"RAM at start: {ram.used/1e9:.1f} / {ram.total/1e9:.1f} GB ({ram.percent}%)")
if ram.percent > 70:
    print("[WARN] RAM usage is high — consider closing other apps.")

try:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
except:
    pass

for _ in range(5): gc.collect()
print("[OK] Cleanup done.\n")

# ============================================================
# CONFIG
# ============================================================
# --- DEMO (fits local GPU, ~2 hours) ---
NUM_SEEDS = 1
MAX_ROUNDS = 4
ENSEMBLE_K = 3
# --- MedIA (needs strong GPU, ~60 GPU-hours) ---
# NUM_SEEDS = 5
# MAX_ROUNDS = 8
# ENSEMBLE_K = 5

HARD_CLASSES = [4, 5, 6]
INITIAL_BUDGET = 20
QUERY_BATCH = 20
RANDOM_STATE = 42
NUM_CLASSES = 7
MEDSAM_SIZE = 1024
BOX_NOISE_PX = 25
BOX_EXPAND_PX = 10

ENCODER_BATCH = 2
TRAIN_BATCH = 16
TRAIN_EPOCHS = 20
LR = 1e-4
ACQ_SUBSAMPLE = 50
RAM_LOG_EVERY = 50

METHODS = ['random', 'entropy', 'variance_only', 'gradient_only', 'boundary_weighted']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"Config: SEEDS={NUM_SEEDS}, ROUNDS={MAX_ROUNDS}, K={ENSEMBLE_K}\n")

# ============================================================
# 1. LOAD PAIRS DIRECTLY FROM DISK
# ============================================================
images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
labels = sorted(glob.glob(os.path.join(LABEL_DIR, "*.png")))
print(f"Images found: {len(images)}")
print(f"Labels found: {len(labels)}")

image_dict = {os.path.basename(f): f for f in images}
label_dict = {os.path.basename(f): f for f in labels}
pairs = [(image_dict[n], label_dict[n]) for n in image_dict if n in label_dict]
print(f"Total matched pairs: {len(pairs)}")

if len(pairs) == 0:
    raise RuntimeError("No matched image-label pairs found.")

first_mask = np.array(Image.open(pairs[0][1]).convert("L"))
print(f"Classes in first mask: {np.unique(first_mask)}\n")

# ============================================================
# 2. PATIENT-WISE SPLIT
# ============================================================
def pid(fname): return os.path.basename(fname).split('_')[0]

patient_map = {}
for img, lbl in pairs:
    patient_map.setdefault(pid(img), []).append((img, lbl))

p_ids = list(patient_map.keys())
tr_val_p, te_p = train_test_split(p_ids, test_size=0.2, random_state=RANDOM_STATE)
tr_p, val_p = train_test_split(tr_val_p, test_size=0.15, random_state=RANDOM_STATE)

train_pool = [p for k in tr_p  for p in patient_map[k]]
val_pairs  = [p for k in val_p for p in patient_map[k]]
test_pairs = [p for k in te_p  for p in patient_map[k]]
print(f"Train: {len(train_pool)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}\n")

# ============================================================
# 3. LOAD MedSAM
# ============================================================
print("Loading MedSAM...")
sam_model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to(DEVICE)
for p in sam_model.vision_encoder.parameters():
    p.requires_grad = False
sam_model.vision_encoder = sam_model.vision_encoder.half()
print(f"Trainable: {sum(p.numel() for p in sam_model.parameters() if p.requires_grad)/1e6:.2f}M\n")

# ============================================================
# 4. UTILITIES
# ============================================================
def load_image_1024(path):
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_pil = Image.fromarray((img * 255).astype(np.uint8)).resize(
        (MEDSAM_SIZE, MEDSAM_SIZE), Image.BILINEAR)
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    return np.stack([arr, arr, arr], 0)

def load_mask_1024(path):
    m = Image.open(path).convert("L").resize((MEDSAM_SIZE, MEDSAM_SIZE), Image.NEAREST)
    return np.array(m, dtype=np.uint8)

def mask_to_box(mask_2d, class_id):
    ys, xs = np.where(mask_2d == class_id)
    if len(ys) == 0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def simulate_user_box(gt_box, noise=BOX_NOISE_PX, expand=BOX_EXPAND_PX, img_size=MEDSAM_SIZE):
    x0, y0, x1, y1 = gt_box
    x0 = x0 + np.random.randint(-noise, noise+1) - expand
    y0 = y0 + np.random.randint(-noise, noise+1) - expand
    x1 = x1 + np.random.randint(-noise, noise+1) + expand
    y1 = y1 + np.random.randint(-noise, noise+1) + expand
    x0 = max(0, min(x0, img_size-1)); y0 = max(0, min(y0, img_size-1))
    x1 = max(0, min(x1, img_size-1)); y1 = max(0, min(y1, img_size-1))
    if x1 <= x0: x1 = x0 + 5
    if y1 <= y0: y1 = y0 + 5
    return [x0, y0, x1, y1]

# ============================================================
# 5. EMBEDDING PRE-COMPUTATION
# ============================================================
print(f"Pre-computing embeddings (batch_size={ENCODER_BATCH})...")

def compute_embeddings_batched(pair_list, batch_size=ENCODER_BATCH, tag="train"):
    sam_model.vision_encoder.eval()
    out = []
    n = len(pair_list)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_imgs = np.stack([load_image_1024(pair_list[i][0]) for i in range(start, end)], 0)
            batch_t = torch.from_numpy(batch_imgs).float().to(DEVICE).half()
            with autocast(device_type='cuda', dtype=torch.float16):
                embs = sam_model.vision_encoder(batch_t).last_hidden_state
            for j in range(start, end):
                _, lp = pair_list[j]
                out.append({
                    'embedding': embs[j-start:j-start+1].detach().cpu().half(),
                    'mask': load_mask_1024(lp),
                    'img_path': pair_list[j][0],
                })
            del batch_t, embs, batch_imgs
            torch.cuda.empty_cache(); gc.collect()
            if end % RAM_LOG_EVERY == 0 or end == n:
                r = psutil.virtual_memory()
                print(f"  [{tag}] {end}/{n}  RAM: {r.used/1e9:.1f}/{r.total/1e9:.1f}GB ({r.percent}%)")
    return out

emb_train_all = compute_embeddings_batched(train_pool, tag="train")
gc.collect()
emb_val = compute_embeddings_batched(val_pairs, tag="val")
gc.collect()
emb_test = compute_embeddings_batched(test_pairs, tag="test")
gc.collect()

r = psutil.virtual_memory()
print(f"Embeddings ready: train={len(emb_train_all)}, val={len(emb_val)}, test={len(emb_test)}")
print(f"RAM after embeddings: {r.used/1e9:.1f}/{r.total/1e9:.1f} GB ({r.percent}%)\n")

# ============================================================
# 6. TRAINABLE WRAPPER
# ============================================================
class MedSAMPromptDecoder(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.sam_model = sam_model
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder

    def get_image_positional_embeddings(self, batch_size):
        if hasattr(self.sam_model, 'get_image_wide_positional_embeddings'):
            pe = self.sam_model.get_image_wide_positional_embeddings()
        else:
            size = self.prompt_encoder.image_embedding_size
            shared = (getattr(self.sam_model, 'shared_image_embedding', None)
                      or getattr(self.prompt_encoder, 'shared_image_embedding', None)
                      or getattr(self.prompt_encoder, 'shared_embedding', None))
            if shared is None:
                for name, module in self.prompt_encoder.named_modules():
                    if 'positional_embedding' in name and hasattr(module, 'positional_embedding'):
                        shared = module; break
            device = next(self.prompt_encoder.parameters()).device
            grid = torch.ones(size, device=device, dtype=torch.float32)
            y_embed = grid.cumsum(dim=0) - 0.5
            x_embed = grid.cumsum(dim=1) - 0.5
            y_embed = y_embed / size[0]; x_embed = x_embed / size[1]
            coords = torch.stack([x_embed, y_embed], dim=-1)
            pe = shared(coords.unsqueeze(0)).permute(0, 3, 1, 2)
        if pe.shape[0] == 1 and batch_size > 1:
            pe = pe.repeat(batch_size, 1, 1, 1)
        elif pe.shape[0] != batch_size:
            pe = pe.expand(batch_size, -1, -1, -1)
        return pe

    def forward(self, image_embedding, input_boxes):
        sparse_emb, dense_emb = self.prompt_encoder(
            input_points=None, input_labels=None,
            input_boxes=input_boxes, input_masks=None,
        )
        batch_size = image_embedding.shape[0]
        image_pe = self.get_image_positional_embeddings(batch_size).to(image_embedding.dtype)
        decoder_out = self.mask_decoder(
            image_embeddings=image_embedding,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        low_res_masks = (decoder_out[0] if isinstance(decoder_out, tuple)
                         else decoder_out.pred_masks if hasattr(decoder_out, 'pred_masks')
                         else decoder_out[0])
        if low_res_masks.dim() == 5: low_res_masks = low_res_masks.squeeze(2)
        if low_res_masks.dim() == 4 and low_res_masks.shape[1] > 1:
            low_res_masks = low_res_masks[:, :1]
        if low_res_masks.dim() == 3: low_res_masks = low_res_masks.unsqueeze(1)
        if low_res_masks.dim() == 5:
            B = low_res_masks.shape[0]
            low_res_masks = low_res_masks.view(B, 1, low_res_masks.shape[-2], low_res_masks.shape[-1])
        return F.interpolate(low_res_masks, size=(MEDSAM_SIZE, MEDSAM_SIZE),
                             mode='bilinear', align_corners=False)

# ============================================================
# 7. LOSSES
# ============================================================
def dice_loss(logits, target, smooth=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1,2,3))
    union = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return 1 - ((2*inter + smooth) / (union + smooth)).mean()

def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)

# ============================================================
# 8. TRAINING
# ============================================================
def make_batches(labeled_embs, batch_size=TRAIN_BATCH):
    samples = []
    for item in labeled_embs:
        mask = item['mask']
        present = [c for c in range(1, NUM_CLASSES) if (mask == c).sum() > 0]
        for c in present:
            gt_box = mask_to_box(mask, c)
            if gt_box is None: continue
            samples.append((item['embedding'], gt_box, (mask == c).astype(np.float32), c))
    random.shuffle(samples)
    return [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]

def train_decoder(model, labeled_embs, epochs=TRAIN_EPOCHS, lr=LR):
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=lr, weight_decay=1e-4)
    scaler = GradScaler()
    model.train()
    for ep in range(epochs):
        for batch in make_batches(labeled_embs):
            embs = torch.cat([b[0] for b in batch], dim=0).to(DEVICE)
            boxes = torch.tensor([[simulate_user_box(b[1])] for b in batch],
                                 dtype=torch.float32).to(DEVICE)
            targets = torch.tensor(np.stack([b[2] for b in batch], 0)
                                   ).unsqueeze(1).to(DEVICE)
            opt.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(embs, boxes)
                loss = bce_loss(logits, targets) + dice_loss(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    return model

# ============================================================
# 9. TRAIN ENSEMBLE
# ============================================================
def train_ensemble(labeled_embs, K=ENSEMBLE_K, base_seed=0):
    models = []
    for k in range(K):
        torch.manual_seed(base_seed * 100 + k)
        np.random.seed(base_seed * 100 + k)
        m = MedSAMPromptDecoder(sam_model).to(DEVICE)
        train_decoder(m, labeled_embs, epochs=TRAIN_EPOCHS, lr=LR)
        models.append(m)
        torch.cuda.empty_cache(); gc.collect()
    return models

# ============================================================
# 10. EVALUATION
# ============================================================
def evaluate(model, emb_list, batch_size=TRAIN_BATCH):
    model.eval()
    per_dice = {c: [] for c in range(1, NUM_CLASSES)}
    samples = []
    for item in emb_list:
        mask = item['mask']
        for c in range(1, NUM_CLASSES):
            if (mask == c).sum() == 0: continue
            gt_box = mask_to_box(mask, c)
            if gt_box is None: continue
            samples.append((item['embedding'], gt_box, (mask == c).astype(np.uint8), c))
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start+batch_size]
            embs = torch.cat([b[0] for b in batch], dim=0).to(DEVICE)
            boxes = torch.tensor([[simulate_user_box(b[1])] for b in batch],
                                 dtype=torch.float32).to(DEVICE)
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(embs, boxes)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze(1).astype(np.uint8)
            for i, (_, _, gt, c) in enumerate(batch):
                pred = preds[i]
                inter = np.logical_and(pred, gt).sum()
                union = pred.sum() + gt.sum()
                dice = 2*inter / union if union > 0 else np.nan
                if not np.isnan(dice): per_dice[c].append(dice)
    macro_dice = np.nanmean([np.mean(v) for v in per_dice.values() if v])
    return macro_dice, per_dice

def hard_dice(per_dice):
    vals = [np.mean(per_dice[c]) for c in HARD_CLASSES if c in per_dice and per_dice[c]]
    return float(np.mean(vals)) if vals else 0.0

# ============================================================
# 11. BAYESIAN ACQUISITION FUNCTION
# ============================================================
def _predict_ensemble_probs(models, emb, box, c):
    emb_t = emb.to(DEVICE)
    box_t = torch.tensor([[box]], dtype=torch.float32).to(DEVICE)
    probs = []
    for m in models:
        m.eval()
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            p = torch.sigmoid(m(emb_t, box_t)).cpu().float().numpy().squeeze()
        probs.append(p)
    return np.stack(probs, 0)

def _acquisition_score(models, item, alpha, beta):
    mask = item['mask']
    present = [c for c in range(1, NUM_CLASSES) if (mask == c).sum() > 0]
    if not present: return 0.0
    img_scores = []
    for c in present:
        gt_box = mask_to_box(mask, c)
        if gt_box is None: continue
        box = simulate_user_box(gt_box)
        probs = _predict_ensemble_probs(models, item['embedding'], box, c)
        variance = probs.var(axis=0)
        mean_pred = probs.mean(axis=0)
        gy, gx = np.gradient(mean_pred)
        grad_mag = np.hypot(gy, gx)
        score = (variance ** alpha) * (grad_mag ** beta)
        img_scores.append(float(score.sum()))
    return float(np.mean(img_scores)) if img_scores else 0.0

def acq_random(pool, models):
    return np.random.choice(len(pool), QUERY_BATCH, replace=False).tolist()

def _acq_wrapper(pool, models, alpha, beta):
    if len(pool) > ACQ_SUBSAMPLE:
        idx_sub = np.random.choice(len(pool), ACQ_SUBSAMPLE, replace=False)
        pool_sub = [pool[i] for i in idx_sub]
    else:
        idx_sub = np.arange(len(pool)); pool_sub = pool
    scores = [_acquisition_score(models, it, alpha=alpha, beta=beta) for it in pool_sub]
    return [int(idx_sub[i]) for i in np.argsort(scores)[-QUERY_BATCH:]]

def acq_variance_only(pool, models): return _acq_wrapper(pool, models, 1.0, 0.0)
def acq_gradient_only(pool, models): return _acq_wrapper(pool, models, 0.0, 1.0)
def acq_boundary_weighted(pool, models): return _acq_wrapper(pool, models, 1.0, 1.0)

def acq_entropy(pool, models):
    if len(pool) > ACQ_SUBSAMPLE:
        idx_sub = np.random.choice(len(pool), ACQ_SUBSAMPLE, replace=False)
        pool_sub = [pool[i] for i in idx_sub]
    else:
        idx_sub = np.arange(len(pool)); pool_sub = pool
    scores = []
    for item in pool_sub:
        mask = item['mask']
        present = [c for c in range(1, NUM_CLASSES) if (mask == c).sum() > 0]
        vals = []
        for c in present:
            gt_box = mask_to_box(mask, c)
            if gt_box is None: continue
            box = simulate_user_box(gt_box)
            probs = _predict_ensemble_probs(models, item['embedding'], box, c)
            p = probs.mean(axis=0).clip(1e-6, 1-1e-6)
            vals.append(float(-(p*np.log(p) + (1-p)*np.log(1-p)).mean()))
        scores.append(np.mean(vals) if vals else 0.0)
    return [int(idx_sub[i]) for i in np.argsort(scores)[-QUERY_BATCH:]]

ACQ_FUNCS = {
    'random': acq_random,
    'entropy': acq_entropy,
    'variance_only': acq_variance_only,
    'gradient_only': acq_gradient_only,
    'boundary_weighted': acq_boundary_weighted,
}

# ============================================================
# 12. ACTIVE LOOP
# ============================================================
def _ckpt_path(method, seed):
    return os.path.join(CKPT_ROOT, f"result_{method}_s{seed}.json")

def _load_ckpt(method, seed):
    p = _ckpt_path(method, seed)
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

def run_al(method, seed):
    cached = _load_ckpt(method, seed)
    if cached is not None:
        print(f"\n=== {method}, seed {seed} === (cached)")
        return cached

    print(f"\n=== {method}, seed {seed} ===")
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    pool = list(emb_train_all); random.shuffle(pool)
    labeled = pool[:INITIAL_BUDGET]
    unlabeled = pool[INITIAL_BUDGET:]
    rows = []

    for r in range(1, MAX_ROUNDS + 1):
        print(f"  Round {r}/{MAX_ROUNDS} (labeled={len(labeled)})")
        models = train_ensemble(labeled, K=ENSEMBLE_K, base_seed=seed)
        md, pcd = evaluate(models[0], emb_test)
        hd = hard_dice(pcd)
        print(f"  Full Dice={md:.4f}  Hard Dice={hd:.4f}")

        torch.save({'seed': seed, 'method': method, 'round': r,
                    'state': models[0].state_dict(), 'hard_dice': hd},
                   os.path.join(CKPT_ROOT, f"model_{method}_s{seed}_r{r}.pth"))

        rows.append({'seed': seed, 'method': method, 'round': r,
                     'labeled': len(labeled), 'full_dice': md,
                     'hard_dice': hd, 'class': None})

        if not unlabeled: break
        idx = ACQ_FUNCS[method](unlabeled, models)
        labeled += [unlabeled[i] for i in idx]
        unlabeled = [u for i, u in enumerate(unlabeled) if i not in idx]

        for m in models: del m
        del models
        torch.cuda.empty_cache(); gc.collect()

    with open(_ckpt_path(method, seed), 'w') as f:
        json.dump(rows, f)
    print(f"  Saved: {method} seed {seed}")
    return rows

# ============================================================
# 13. FULL-DATA BASELINE
# ============================================================
def run_full():
    print("\n=== Full-data baseline ===")
    ckpt = os.path.join(CKPT_ROOT, "full_baseline.json")
    if os.path.exists(ckpt):
        with open(ckpt) as f: cached = json.load(f)
        print(f"  Cached: Dice={cached['full_dice']:.4f}  Hard={cached['hard_dice']:.4f}")
        return cached['full_dice'], cached['hard_dice']

    np.random.seed(0); torch.manual_seed(0)
    model = MedSAMPromptDecoder(sam_model).to(DEVICE)
    train_decoder(model, emb_train_all, epochs=25, lr=LR)
    md, pcd = evaluate(model, emb_test)
    hd = hard_dice(pcd)
    print(f"Full: Dice={md:.4f}  Hard={hd:.4f}")
    with open(ckpt, 'w') as f:
        json.dump({'full_dice': md, 'hard_dice': hd}, f)
    return md, hd

# ============================================================
# 14. RUN EVERYTHING
# ============================================================
all_rows = []
for seed in [RANDOM_STATE + i for i in range(NUM_SEEDS)]:
    for method in METHODS:
        all_rows += run_al(method, seed)

full_dice, full_hard = run_full()
df = pd.DataFrame(all_rows)
out_csv = os.path.join(OUTPUT_DIR, "medsam_media_results.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved results: {out_csv}")

# Free memory
try: del emb_train_all
except: pass
try: del emb_val
except: pass
torch.cuda.empty_cache(); gc.collect()

# ============================================================
# 15. STATISTICS
# ============================================================
macro = df[df['class'].isna()].copy()
print("\n=== Hard-organ Dice by method ===")
print(macro.groupby(['method', 'labeled'])['hard_dice'].agg(['mean','std']).round(4))

final_budget = macro['labeled'].max()
final = macro[macro['labeled'] == final_budget]
print("\n=== Final round summary ===")
for m in METHODS:
    vals = final[final.method == m]['hard_dice'].values
    print(f"  {m:20s}: {vals.mean():.4f} +/- {vals.std():.4f}  (n={len(vals)})")

if NUM_SEEDS > 1:
    print("\n=== Paired t-tests vs random ===")
    r_final = final[final.method == 'random']['hard_dice'].values
    for m in METHODS:
        if m == 'random': continue
        t_final = final[final.method == m]['hard_dice'].values
        if len(r_final) > 1 and len(t_final) == len(r_final):
            _, p = ttest_rel(t_final, r_final)
            print(f"  {m:20s} vs random: p={p:.4f}")

# ============================================================
# 16. FIGURES
# ============================================================
METHOD_COLORS = {
    'random': '#888888', 'entropy': '#3498db',
    'variance_only': '#f39c12', 'gradient_only': '#9b59b6',
    'boundary_weighted': '#e74c3c',
}
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (metric, title) in enumerate([('full_dice', 'All-classes Dice'),
                                        ('hard_dice', 'Hard-organ Dice')]):
    ax = axes.flat[idx]
    for m in METHODS:
        sub = macro[macro['method'] == m].groupby('labeled')[metric]
        mean, std = sub.mean(), sub.std()
        ax.plot(mean.index, mean, marker='o', color=METHOD_COLORS[m], label=m, linewidth=2)
        if NUM_SEEDS > 1:
            ax.fill_between(mean.index, mean-std, mean+std, alpha=0.15, color=METHOD_COLORS[m])
    ax.set_xlabel('Labelled slices'); ax.set_ylabel(title)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "learning_curves.png")
plt.savefig(fig_path); plt.close()
print(f"\nSaved figure: {fig_path}")

# ============================================================
# 17. DONE
# ============================================================
print("\n" + "="*60)
print("DONE.")
print(f"Results CSV    : {out_csv}")
print(f"Figures        : {FIGURES_DIR}")
print(f"Checkpoints    : {CKPT_ROOT}")
print("="*60)
