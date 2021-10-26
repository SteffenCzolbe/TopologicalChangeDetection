# Spot the difference: Detection of Topological Changes via Geometric Alignment

Steffen Czolbe, Aasa Feragen, Oswin Krause

[[NeurIPS 2021 Paper]](https://openreview.net/forum?id=a-Lbgfy9RqV)
[[NeurIPS 2021 Presentation]](https://youtu.be/MQ4GvM0up5c)
[[NeurIPS 2021 Poster]](TBA)

[![Video](https://img.youtube.com/vi/MQ4GvM0up5c/hqdefault.jpg)](https://youtu.be/MQ4GvM0up5c)

This repository contains all experiments presented in the paper, the code used to generate the figures, and instructions and scripts to re-produce all results. Implementation in the deep-learning framework pytorch.

# Publications

Accepted for NeurIPS 2021 (26% acceptence rate). cite as:

```
@inproceedings{czolbe2021topology,
    title={Spot the Difference: Detection of Topological Changes via Geometric Alignment},
    author={Czolbe, Steffen and Feragen, Aasa and Krause, Oswin},
    booktitle={Advances in Neural Information Processing Systems},
    volume={34},
    year={2021}
}
```

# Prerequisites

## Dependencies

All dependencies are listed the the file `requirements.txt`. Simply install them with a package manager of your choice, eg.

```
$ pip3 install -r requirements.txt
```

## Data

This work relies on the three datasets used in the paper: Platelet-EM, Brain-MRI, and BraTS.

We are sadly not allowed to re-distribute this data, but give instructions on how this data is structured for this codebase.
to use your own data, you will have to implement the dataloaders present in `src/datamodules/` for your dataset.

### Platelet-EM Dataset

We use the Platelet-EM dataset from Quay et al. (2018), with topology-change annotations of the test and validation sets by us. The topology change annotations are included within this repository. The original images and segmentations have to be placed in:

```
deepsimreg/
    data/
        platelet_em/
            raw/
                images/
                    24-images.tif
                    50-images.tif
                labels-class/
                    24-class.tif
                    50-class.tif
                labels-semantic/
                    24-semantic.tif
                    50-semantic.tif
```

afterwards, run the preprocessing script:

```
$ pyhon3 -m data.platelet_em.preprocess
```

### Brain-MRI Dataset

The Brain-MRI scans have been taken from the publically accessible [ABIDEI, ABIDEII](http://fcon_1000.projects.nitrc.org/indi/abide/), [OASIS3](https://www.oasis-brains.org/) studies. We used Freesurfer and some custom scripts to perform the preprocessing steps of

- intensity normalization
- skullstripping
- affine alignment
- automatic segmentation
- crop to 160x192x224
- Segmentation areas of LH and RH combined to single labels
- some smaller segmentations removed/combined, Total 22 classes left

The resulting intensity and label volumes are then organized in a separate directory:

```
this_repository/
    <you are here>
brain_mris/
    data/
        <subject_id>/
            "brain_aligned.nii.gz"
            "seg_coalesced_aligned.nii.gz"
        <subject_id>/
            "brain_aligned.nii.gz"
            "seg_coalesced_aligned.nii.gz"
        ...
        metadata.csv
```

The `metadata.csv` has to contain a tabular listing of all subjects, with columns `subject_id` and `SPLIT`. Valid values for the split-column are `train`, `val`, `test`.

### BraTS

The BraTS dataset have been taken from the publically accessible [BraTS2020 challenge](https://www.med.upenn.edu/cbica/brats2020/data.html). We used Freesurfer to perform the preprocessing steps of

- intensity normalization
- affine alignment
- crop to 160x192x224

The resulting intensity and label volumes are then organized in a separate directory:

```
this_repository/
    <you are here>
BraTS/
    preprocessed_data/
        <subject_id>/
            "t1_aligned_normalized.nii.gz"
            "seg_aligned.nii.gz"
        <subject_id>/
            "t1_aligned_normalized.nii.gz"
            "seg_aligned.nii.gz"
        ...
        metadata.csv
```

The `metadata.csv` has to contain a tabular listing of all subjects, with columns `subject_id`, `SPLIT` (only accepted value is `train`, as only the BraTS train set comes with annotations), `AUTO_PROCESSING` (only accepted value is `OK`) and `center_slice_tumor_size` (values smaller 500 are filtered out).

# Reproduction of the paper results

We provide convenient scripts for training and evaluating all models and baselines.

Train all models with:

```
$ ./scripts/train_seg_models.sh
$ ./scripts/train_reg_models.sh
$ ./scripts/train_baselines.sh
```

After each training, the saved model weight have to be moved to the `weights/` directory for further analysis. The corresponding `mv` commands are included in the training scripts, but commented out for parallel cluster training.

Once all models have been moved to thir locaions in the `weights/` directory, all experiments and evaluations as well as figures from the paper can be re-generated with

```
$ ./scripts/plot.sh
```
