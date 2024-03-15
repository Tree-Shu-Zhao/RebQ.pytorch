# Reconstruct before Query: Continual Missing Modality Learning with Decomposed Prompt Collaboration

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
