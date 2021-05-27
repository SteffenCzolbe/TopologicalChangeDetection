# Spot the difference: Topological anomaly detection via geometric alignment

More detailed instructions will be released on publication.

# Prerequisites

Install dependencies listed in `requirements.txt`

As we are not allowed to re-distribute data, it is nessesary to supply the data yourself.

### Brain-MRI

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
        ...
        metadata.csv
```

The `metadata.csv` has to contain a tabular listing of all subjects, with columns `subject_id`, `SPLIT` (only accepted value is `train`, as only the BraTS train set comes with annotations), `AUTO_PROCESSING` (only accepted value is `OK`) and `center_slice_tumor_size` (values smaller 500 are filtered out).

# Re-run experiments

Train all models with:

```
$ ./scripts/train.sh
```

Run evaluation and re-generate all figures from the paper with

```
$ ./scripts/plot.sh
```
