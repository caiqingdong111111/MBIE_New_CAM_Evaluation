## MBIE: A Novel Evaluation Metric for Class Activation Mapping Methods

Official implementation of the evaluation metric proposed in:

> **A Novel Evaluation Metric for Class Activation Mapping Methods in Weakly Supervised Learning**

This repository provides the implementation of **MBIE**, an evaluation metric designed to assess the quality of Class Activation Mapping (CAM) methods for weakly supervised learning, particularly in the weakly supervised semantic segmentation.

## Overview

Class Activation Mapping (CAM) methods are widely used in weakly supervised learning tasks to identify image regions associated with target classes.

MBIE evaluates CAMs by considering both:

* **Activation within the target object regions**, measuring how strongly a CAM responds to the corresponding object; and
* **Activation leakage into the background**, measuring undesirable activation outside the target object.

The final MBIE score balances target-region activation against background activation leakage.

A higher MBIE score indicates that the CAM provides stronger activation within the target object regions while producing less activation leakage into the background.

## Repository Structure

```text
MBIE_New_CAM_Evaluation/
│
├── CAM_result_samples/
│   └── ...                     # Example CAM results (.npy)
│
├── VOC2012/
│   ├── SegmentationClass/      # Ground-truth segmentation masks
│   ├── train.txt               # Image list
│   └── cls_labels.npy          # Image-level class labels
│
├── MBIE.py                     # MBIE evaluation script
├── LICENSE
└── README.md
```

## Requirements

The implementation requires Python and the following packages:

```bash
pip install numpy pillow
```

## Data Preparation

The current implementation uses the **PASCAL VOC 2012** dataset.

Ground-truth segmentation masks should be placed in:

```text
VOC2012/SegmentationClass/
```

The image list is expected at:

```text
VOC2012/train.txt
```

Image-level class information is loaded from:

```text
VOC2012/cls_labels.npy
```

## Preparing CAM Results

CAM results are stored in:

```text
CAM_result_samples/
```

For each image in `train.txt`, the corresponding CAM result are saved as:

```text
CAM_result_samples/<image_id>.npy
```

Each `.npy` file should contain a Python dictionary in which the keys represent class indices and the corresponding values contain the CAMs for those classes.

Example:

```python
{
    0: cam_for_class_0,
    7: cam_for_class_7
}
```

The spatial dimensions of each CAM should correspond to the ground-truth segmentation mask used for evaluation.

## Usage

After preparing the dataset and CAM results, run:

```bash
python MBIE.py
```

The script reports:

```text
MBIEc
mMBIEc
MBIEcb
MBIE
```

where:

* **MBIEc** represents the class-wise activation within the corresponding object regions.
* **mMBIEc** is the mean object-region activation across classes.
* **MBIEcb** represents activation leakage into the background.
* **MBIE** is the final evaluation score.

The final score is computed as:

```text
MBIE = max(mMBIEc - MBIEcb, 0)
```

## Example CAM Results

Example CAM results are provided in:

```text
CAM_result_samples/
```

These samples demonstrate the expected format of the CAM files used by `MBIE.py`.

## Citation

If you find this repository or the proposed evaluation metric useful in your research, please consider citing our paper.

```bibtex
@article{cai_mbie,
  title   = {A Novel Evaluation Metric for Class Activation Mapping Methods in Weakly Supervised Learning},
  author  = {Cai, Qingdong and Abhayaratne, Charith},
  note    = {Citation information will be updated after publication}
}
```

> **Note:** The paper citation and DOI will be updated once the final publication information becomes available.

## Status

The implementation is publicly available. Documentation and citation information will be updated alongside the publication status of the corresponding paper.

## License

This project is released under the MIT License. See `LICENSE` for details.
