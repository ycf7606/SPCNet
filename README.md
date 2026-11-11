# SPCNet

PyTorch implementation of SPCNet: Shared-Pattern Collaborative Network for Multi-Source Remote Sensing Data Classification.

This repository accompanies the paper [“Shared-Pattern Collaborative Network for Multi-Source Remote Sensing Data Classification”](https://ieeexplore.ieee.org/document/11774838), published in the *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* (IEEE JSTARS), 2026.

## 🔎 Overview

SPCNet is designed for multi-source remote sensing image classification, especially the joint interpretation of hyperspectral imagery (HSI) with LiDAR or SAR data. The network combines a Shared-Pattern Collaborative Block (SPCB), which captures patterns shared across modalities and depths, with a Region-Adaptive Modality Allocator (RAMA), which fuses modality features using spatially varying weights.

## 📊 Supported Datasets

The code supports the following dataset settings:

`dataset_type` | Dataset | Modalities | Classes
--- | --- | --- | ---:
`0` | Houston2013 | HSI + LiDAR | 15
`1` | Houston2018 | HSI + LiDAR | 20
`2` | Trento | HSI + LiDAR | 6
`3` | Berlin | HSI + SAR | 8
`4` | Augsburg | HSI + SAR | 7

## 📁 Repository Structure

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

## 📦 Data Preparation

Dataset files are not included in this repository. Obtain the datasets from their respective sources and keep the MATLAB filenames and variable names expected by the loader.

```text
workspace/
├── SPCNet/
└── data/
    ├── Houston2013/
    ├── Houston2018/
    ├── Trento/
    ├── Berlin/
    └── Augsburg/
```

The default data root is `SPCNet/data/`. To use a different location, set `HSI_DATA_ROOT` to the directory containing the dataset folders.

| Dataset | Required files |
| --- | --- |
| Houston2013, Houston2018 | `houston_hsi.mat`, `houston_lidar.mat`, `houston_gt.mat`, `houston_index.mat` |
| Trento | `trento_hsi.mat`, `trento_lidar.mat`, `trento_gt.mat`, `trento_index.mat` |
| Berlin | `berlin_hsi.mat`, `berlin_sar.mat`, `berlin_gt.mat`, `berlin_index.mat` |
| Augsburg | `augsburg_hsi.mat`, `augsburg_sar.mat`, `augsburg_gt.mat`, `augsburg_index.mat` |

The MATLAB variables follow the corresponding dataset prefixes. For example, Houston data uses `houston_hsi`, `houston_lidar`, `houston_gt`, `houston_train`, `houston_test`, and `houston_all`.

PowerShell:

```powershell
$env:HSI_DATA_ROOT = "D:\datasets\hsi"
```

Linux or macOS:

```bash
export HSI_DATA_ROOT=/path/to/datasets/hsi
```

## 🧩 Pretrained Weights

Pretrained weights are not included. Train a model using the commands below before running evaluation. Checkpoints are saved under `model/SPCNet/`.

## 🚀 Training and Testing

Use Python 3.10. Create and activate a virtual environment, then install the dependencies.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The included paper configuration is for Houston2013 (`dataset_type = 0`):

```bash
python scripts/train.py --config configs/houston2013.json
```

Training evaluates the test split after each epoch and saves the checkpoint with the highest test OA. To evaluate the saved checkpoint:

```bash
python scripts/test.py --config configs/houston2013.json
```

To train or evaluate another supported dataset, update `dataset_type` and the experiment settings in a configuration file. Dataset-specific class counts, auxiliary-modality dimensions, preprocessing, and report labels are selected by the code.

## 📝 Outputs

The project saves trained models to `model/`, training logs to `log/`, and evaluation reports to `report/`.

## 💡 Notes

The repository includes one configuration, for Houston2013. Other datasets can be selected by editing a configuration. Datasets and generated outputs are excluded from version control. No license file is included, following the reference repository's setup.

## 📚 Citation

If you use this repository, please cite:

```bibtex
@article{yang2026spcnet,
  title   = {Shared-Pattern Collaborative Network for Multi-Source Remote Sensing Data Classification},
  author  = {Yang, Chenfang and Gong, Chuanzheng and Gao, Feng and Dong, Junyu and Du, Qian},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026},
  pages   = {1--15},
  doi     = {10.1109/JSTARS.2026.3750641},
  url     = {https://ieeexplore.ieee.org/document/11774838}
}
```

## 📬 Contact

For questions about the paper or code, please contact:

- Yang Chenfang: [yangchenfang@stu.ouc.edu.cn](mailto:yangchenfang@stu.ouc.edu.cn)
- Feng Gao (Advisor): [gaofeng@ouc.edu.cn](mailto:gaofeng@ouc.edu.cn)
