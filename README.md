# Reconstruct before Query: Continual Missing Modality Learning with Decomposed Prompt Collaboration

This repository contains the code release of RebQ, from our paper: 

[**Reconstruct before Query: Continual Missing Modality Learning with Decomposed Prompt Collaboration**](https://arxiv.org/pdf/2403.11373.pdf)

Shu Zhao, Xiaohan Zou, Tan Yu, Huijuan Xu

Pennsylvania State University, NVIDIA

arXiv:2403.11373, 2024.

If this code and/or paper is useful in your research, please cite:

```bibtex
@article{zhao2024rebq,
  title={Reconstruct before Query: Continual Missing Modality Learning with Decomposed Prompt Collaboration},
  author={Shu Zhao and
          Xiaohan Zou and
          Tan Yu and
          Huijuan Xu},
  journal={arXiv preprint arXiv:2403.11373},
  year={2024}
}
```

## Installing Dependencies

We tested our code on Ubuntu 22.04 with PyTorch 1.13. You can use `environment.yml` and `requirements.txt` to install dependencies.

## Data Preparation

Download `UPMC-Food101` and `MM-IMDb` datasets according to the [MAP](https://github.com/YiLunLee/missing_aware_prompts) repo and organize them as following:
```bash
data
├── MM-IMDB-CMML
│   ├── images
│   ├── labels
│   └── MM-IMDB-CMML.json
└── UPMC-Food101-CMML
    ├── images
    ├── texts
    └── UPMC-Food101-CMML.json
```

## Run

`bash scripts/food101_both_0.7.sh`

## Acknowledgement

1. [Missing-Aware Prompt](https://github.com/YiLunLee/missing_aware_prompts)
2. [A Pre-trained Model-based Continual Learning ToolBox](https://github.com/sun-hailong/LAMDA-PILOT)
