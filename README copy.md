#https://github.com/Gutman-Lab/wsi-tissue-detection/blob/tissue-compartment-output-points/notebooks/tissue-compartment-output-points-comparisons.ipynb



# wsi-tissue-detection
Whole-slide-images (WSI) tissue detection for neuropathology.

### Information
There are two sets of codes in this repository. (1) Code related to training SegFormer model on a binary tissue detection task. The images for this task are tiled images of low resolution (thumbnails) of WSIs. (2) Code related to training SegFormer model on a a multiclass problem of different tissue classes, including gray matter, white matter, and leptomenenges. The scripts for these are numbered for easy understanding of the order to run them in. Also, tutorial notebooks are provided for learning the code used in the scripts.