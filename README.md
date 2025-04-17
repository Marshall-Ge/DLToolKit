## Introduction

This repository provides a toolbox of multimodal (primarily vision-language) models and components.
It aims to assist researchers and engineers in easily building multimodal models using the popular works in the field.
You can also organize or extend these components to create your own models.

In addition to model implementations, it offers a general way for training and evaluating models on commonly used benchmarks.

## Implemented Works

### Models
- [x] [BLIP](models/blip.py)
- [ ] [CLIP]()
- [ ] [LDC]()

### Components
- [x] [Cross Attention](models/attention.py)
- [x] [Cross Attention with Multi-Head](models/attention.py)
- [x] [Vision Transformer](models/attention.py)

### Mechanisms
- [x] [Interpolate position embedding](models/utils.py)

## Environment Setup
Using conda
```bash
conda create -n my_env python=3.11
conda activate my_env
```

```bash
pip install -r requirements.txt
```
Using uv
```bash
uv sync
```

## References
1. [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://proceedings.mlr.press/v162/li22n.html)
2. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
3. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

