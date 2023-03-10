# VGGNet implementation in PyTorch


This is an implementation of VGGNet architecture proposed by Karen Simonyan et al. in the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) using PyTorch. The files contain implementations of both VGG-16 and VGG-19 architectures.

The Jupyter Notebook contains details about the architecture and implementation steps, the Python script contains the code.

The Jupyter Notebook and Python files also contain image pipeline for the Tiny ImageNet dataset, however I did not train the model on the dataset due to hardware limitations. If you wish to train the model using the Tiny ImageNet dataset then you should download it from [Tiny-ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip), I did not include the dataset in the repository as it is quite large, however it is very straight forward to download and train the model after you download it, just mention the file path of the `tiny-imagenet-200` folder in the `DATA_PATH` in `main.py`.


**NOTE:** This repository contains the model architecture of the original paper as proposed by Simonyan et al., the original architecture was trained on the ILSVRC 2014 dataset consisting of 1.2 million images distributed among 1000 classes. While the VGGNet architecture is a good model for image classification tasks, the hyperparameters such as number of activation maps, kernel size, stride, etc must be tuned according to the problem. Also during my limited time using VGG-19 on CIFAR10, the Xavier Glorot initialization suggeted in the paper hindered the ability of the model to learn, you should try Kaiming He initialization if possible.

<div>
<img src="https://cdn.discordapp.com/attachments/418819379174572043/1079830439025451108/1hs8Ud3X2LBzf5XMAFTmGGw.png" width="1100" alt = "Max Ferguson et al">
</div>
