# boundary-weighted-active-learning
Code for "Boundary-Aware Bayesian Active Learning for Annotation-Efficient Abdominal CT Segmentation with Foundation Models". Includes LoRA fine-tuning of MedSAM, five acquisition functions, and figure-generation scripts.
# Boundary-Aware Bayesian Active Learning for Annotation-Efficient Abdominal CT Segmentation

Code for training MedSAM with a LoRA-fine-tuned decoder under a boundary-weighted acquisition criterion for abdominal CT segmentation.

## Overview

This repository provides the full training pipeline, evaluation scripts, and figure-generation code used in the manuscript. The pipeline implements five active learning strategies — random, entropy, variance-only, gradient-only, and boundary-weighted — evaluated across five labelling budgets (20, 40, 60, 80, 100 slices) on a six-organ abdominal CT dataset.

## Dataset

Experiments use the **Abdominal CT Scans** dataset by Daniel Gut.

- Mendeley Data, V1
- DOI: [10.17632/6x684vg2bg.1](https://doi.org/10.17632/6x684vg2bg.1)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset is not redistributed here. Download it directly from Mendeley Data and place the images and labels folders at:
