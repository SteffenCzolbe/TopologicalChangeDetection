# segmentation of brain into GM, WM, background
# example/tutorial: https://dipy.org/documentation/1.0.0./examples_built/tissue_classification/

import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_tissue_data, read_tissue_data
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.io.image import load_nifti
filename = 'abide1.nii.gz'
filename2 = 'abide2.nii.gz'
filename3 = 'brats.nii.gz'
filename4 = 'oasis3.nii.gz'
t1, affine, img = load_nifti(filename3, return_img=True)
hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass, beta)
