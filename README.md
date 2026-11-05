<div align="center">

<h1>Marrying Text-to-Motion Generation<br>with Skeleton-Based Action Recognition</h1>

<p>
  Jidong Kuang &nbsp;&middot;&nbsp; Hongsong Wang<sup>*</sup> &nbsp;&middot;&nbsp; Jie Gui<sup>*</sup>
  &nbsp;&middot;&nbsp; Yuan Yan Tang &nbsp;&middot;&nbsp; James Tin-Yau Kwok
  <br>
  Southeast University &nbsp;&middot;&nbsp; University of Macau &nbsp;&middot;&nbsp; UOW College Hong Kong &nbsp;&middot;&nbsp; HKUST
  <br>
  <sup>*</sup>Corresponding authors
</p>

<p>
  <a href="https://jidongkuang.github.io/projects/CoAMD/">
    <img src="https://img.shields.io/badge/Project-Page-0b70ad.svg?style=flat&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2604.17090">
    <img src="https://img.shields.io/badge/arXiv-2604.17090-b31b1b.svg?logo=arxiv" alt="arXiv">
  </a>
  <a href="https://github.com/jidongkuang/CoAMD">
    <img src="https://img.shields.io/badge/Code-GitHub-181717.svg?style=flat&logo=github&logoColor=white" alt="Code on GitHub">
  </a>
  <img src="https://img.shields.io/github/stars/jidongkuang/CoAMD?style=flat" alt="GitHub stars">
</p>

</div>

This repository is a cleaned, GitHub-ready code release for the main CoAMD experiments in our paper *Marrying Text-to-Motion Generation with Skeleton-Based Action Recognition*.

The release focuses on the core HumanML3D pipeline used in our paper:

- `python -m train.train_ae`: train the 3-stream autoencoder (AE)
- `python -m eval.eval_ae`: evaluate AE reconstruction quality
- `python -m train.train_mar`: train the multi-modal action recognizer (MAR) reward model
- `python -m train.train_coamd`: train the CoAMD motion generator
- `python -m eval.eval_coamd_guided`: evaluate reward-guided text-to-motion generation

Compared with the original research workspace, this folder is reorganized into a standalone repository layout with cleaner module boundaries, config files, and preparation scripts.

---

## 📑 Table of Contents

- [🗂️ Repository Layout](#️-repository-layout)
- [⚙️ Setup](#️-setup)
- [🚀 Quick Start](#-quick-start)
- [📌 Notes](#-notes)
- [🙏 Acknowledgements](#-acknowledgements)
- [📝 Citation](#-citation)

## 🗂️ Repository Layout

```text
CoAMD/
├── assets/                  # Static assets such as label maps
├── coamd/
│   ├── data/                # Dataset loaders
│   ├── diffusions/          # Diffusion / transport backends
│   ├── evaluation/          # Evaluators and metric utilities
│   ├── models/              # CoAMD generator, MAR reward model, AE
│   └── utils/               # Config, logging, motion helpers
├── configs/                 # Example YAML configs
│   └── humanml3d/
├── eval/                    # Evaluation entrypoints
├── prepare/                 # Setup / download scripts
├── train/                   # Training entrypoints
├── environment.yml
├── requirements.txt
└── README.md
```

## ⚙️ Setup

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate coamd
```

### 2. Download evaluator dependencies

```bash
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

### 3. Prepare datasets

Place HumanML3D under `./datasets/HumanML3D/` with at least the following structure:

```text
datasets/
└── HumanML3D/
    ├── annotations_actions_400.json
    ├── mean_std/
    │   └── original_absolute/
    │       ├── Mean.npy
    │       └── Std.npy
    ├── new_joints/
    ├── texts/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### 4. Place pretrained checkpoints

The released scripts expect the following local structure by default:

```text
pretrained/
├── ae/
│   └── latest.tar
├── coamd/
│   └── best_fid.tar
└── mar/
    └── top1_best.tar
```

Update the YAML configs or CLI flags if your checkpoints are stored elsewhere.

## 🚀 Quick Start

### Train AE

```bash
python -m train.train_ae --config configs/humanml3d/train_ae.yaml
```

### Evaluate AE

```bash
python -m eval.eval_ae --config configs/humanml3d/eval_ae.yaml
```

### Train MAR

```bash
python -m train.train_mar --config configs/humanml3d/train_mar.yaml
```

### Train CoAMD

```bash
python -m train.train_coamd --config configs/humanml3d/train_coamd.yaml
```

### Evaluate reward-guided generation

```bash
python -m eval.eval_coamd_guided --config configs/humanml3d/eval_coamd_guided.yaml
```

## 📌 Notes

- This release is intentionally centered on the main HumanML3D experiments used in the paper.
- Several research-only artifacts from the original workspace, such as ad-hoc prompt files, logs, notebooks, and temporary visualization scripts, were removed.
- The codebase is reorganized relative to the original MARDM-based prototype to make the public release easier to navigate and extend.

## 🙏 Acknowledgements

This project builds on several excellent motion-generation and motion-understanding codebases, especially MARDM and the broader HumanML3D evaluation ecosystem. Please also credit the original upstream projects when appropriate.

## 📝 Citation

Please update the citation block below with the final arXiv metadata before the public release.

```bibtex
@misc{kuang2026marryingtexttomotiongenerationskeletonbased,
      title={Marrying Text-to-Motion Generation with Skeleton-Based Action Recognition}, 
      author={Jidong Kuang and Hongsong Wang and Jie Gui},
      year={2026},
      eprint={2604.17090},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.17090}, 
}
```
