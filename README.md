# stereo_rcnn_pytorch
This repository has been created to train stereo rcnn and convert pretrained model from pytorch to tensorflow using onnx library.

0.0. Install Pytorch:

    conda create -n env_stereo python=3.6
    conda activate env_stereo
    conda install pytorch=1.0.0 cuda90 -c pytorch
    conda install torchvision -c pytorch
    
0.1. Other dependencies:

    git clone https://github.com/tomshalini/stereo_rcnn_pytorch.git
    cd stereo_rcnn_pytorch
    pip install -r requirements.txt
    
0.2. Build:

    cd lib
    python setup.py build develop
    cd ..

# Create symlinks:
    
    cd data/kitti
    ln -s stereo_rcnn_pytorch/object object
    cd ../..
    
# Training

    Download the Res-101 pretrained weight https://drive.google.com/file/d/1_t8TtUevtMdnvZ2SoD7Ut8sS1adyCKTt/view, and put it into data/pretrained_model
    
    Set corresponding CUDA_VISIBLE_DEVICES in train.sh, and simply run
    It consumes ~11G GPU memery for training. 
    The trained model and training log are saved in /models_stereo by default.
    
# Testing
    
    Download trained weight https://drive.google.com/uc?id=1rZ5AsMms7-oO-VfoNTAmBFOr8O2L0-xt&export=download and put it into models_stereo/, then just run
    test_net.py
   
# Inference

    run predict.py
