# EP-CSANet
The source code of an article submitted to IEEE Transactions on Very Large Scale Integration (VLSI) Systems


EP-CSANet Image Dehazing
The PyTorch code for the paper "An Efficient Accelerator for Dehazing Neural Network Based on Physical Perception Model and Cross-Scale Pixel Attention."

1.Dependencies and Installation:
Python 3.9
PyTorch >= 1.0
NVIDIA GPU+CUDA
Numpy

2.Dataset Preparation：
Create a folder named "data". Then, download the dataset from the following link: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html, and place it in the "data" directory.
3.Training Steps：
python train.py
The script will automatically dump some validation results into the "samples" folder after each epoch. Model snapshots will be dumped into the "snapshots" folder.
4.Testing Steps：
python test.py
The script takes images from the "test_images" folder and dumps the dehazed images into the "dehaze_results" folder.
