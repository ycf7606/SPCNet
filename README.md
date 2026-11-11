# SPCNet

PyTorch implementation of the Shared-Pattern Collaborative Network (SPCNet) for multi-source remote sensing image classification.

This repository accompanies “Shared-Pattern Collaborative Network for Multi-Source Remote Sensing Data Classification,” accepted for publication in the *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* (2026). It contains the SPCNet model, training and evaluation code, and the Houston2013 configuration.

## Overview

SPCNet processes hyperspectral imagery (HSI) together with a registered auxiliary modality. Its Shared-Pattern Collaborative Block (SPCB) combines a large-kernel operator shared across modalities and depths with an input-conditioned dynamic convolution branch. The Region-Adaptive Modality Allocator (RAMA) fuses the modality features using spatially varying weights.

## Supported datasets

| Dataset | Modalities | Classes | `dataset_type` |
| --- | --- | ---: | ---: |
| Houston2013 | HSI + LiDAR | 15 | 0 |

## Repository structure

```text
SPCNet/
├── configs/
│   └── houston2013.json
├── scripts/
│   ├── train.py
│   └── test.py
├── src/
│   ├── models/spcnet/
│   ├── config.py
│   ├── data.py
│   ├── model_registry.py
│   ├── runner.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── metrics.py
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

## Data preparation

Dataset files are not included. Obtain Houston2013 and use the same train/test partition as the paper. The loader expects the following files and MATLAB variable names:

| Dataset directory | Files | MATLAB variables |
| --- | --- | --- |
| `Houston2013/` | `houston_hsi.mat`, `houston_lidar.mat`, `houston_gt.mat`, `houston_index.mat` | `houston_hsi`, `houston_lidar`, `houston_gt`, `houston_train`, `houston_test`, `houston_all` |

By default, place `Houston2013/` under `data/` in the repository root. To store the data elsewhere, set `HSI_DATA_ROOT` to the directory containing `Houston2013/`.

PowerShell:

```powershell
$env:HSI_DATA_ROOT = "D:\datasets\hsi"
```

Linux or macOS:

```bash
export HSI_DATA_ROOT=/path/to/datasets/hsi
```

## Installation

Use Python 3.10 and a PyTorch build compatible with the installed CUDA driver.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Pretrained weights

Pretrained weights are not included. Run training with a dataset configuration before using the evaluation command. Checkpoints are saved under `model/SPCNet/`.

## Training and evaluation

The provided configuration records the Houston2013 preprocessing and SPCNet settings described in the paper.

```bash
python scripts/train.py --config configs/houston2013.json
```

Training evaluates the test split after each epoch, saves the checkpoint with the highest test OA, and writes a report. To evaluate that checkpoint later, pass the same configuration to the test script:

```bash
python scripts/test.py --config configs/houston2013.json
```

The main settings are:

| Dataset | PCA channels | Patch size | Shared-LK size | SPCB depth | Epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Houston2013 | 30 | 11 | 11 | 8 | 35 |

The configuration uses Adam, learning rate `2e-4`, weight decay `0`, batch size `128`, evaluation batch size `512`, dropout `0.05`, and random seed `6`.

## Outputs

Generated checkpoints, logs, and classification reports are saved under `model/SPCNet/`, `log/SPCNet/`, and `report/SPCNet/`. These generated files and datasets are excluded from version control.

## Citation

```bibtex
@article{yang2026spcnet,
  title   = {Shared-Pattern Collaborative Network for Multi-Source Remote Sensing Data Classification},
  author  = {Yang, Chenfang and Gong, Chuanzheng and Gao, Feng and Dong, Junyu and Du, Qian},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026},
  note    = {Accepted for publication}
}
```

## Contact

For questions about the paper or code, contact Feng Gao at [gaofeng@ouc.edu.cn](mailto:gaofeng@ouc.edu.cn).

## License

No license file is included, following the current setup of the reference RSCNet repository. Public availability does not grant additional permission to reuse, modify, or redistribute this code. Contact Feng Gao regarding reuse permissions.
