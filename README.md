## Image Captioning
A Python Convolutional-Recurrent Nueral Network Trained for Image Classification

This repository contains files created in fullfilment of the <strong>Nvidia-sponsored Udacity Nanodegree in Computer Vision</strong>.  To satisfy the requirements of the Image Captioning assignment, I created a CNN-RNN architecture that took in as inputs the images and captions of Microsoft's COCO dataset, a popular API used for training computer vision neural networks.  The output for my CNN-RNN model is a randomly selected image with an accompanying caption created by my neural net.

<u>A short description of each file in the repository follows:</u>

<em>Dataset_Git.ipnyb</em> is a Jupyter Notebook demonstrating the creation of the COCO training dataset.

<em>Image_Cap_Training_Git</em> is a notebook showing the training process through which I put my CNN-RNN architecture.

<em>Image_Cap_Test_Git</em> is a notebook in which I show the accuracy of my model to autonomously generate captions for random images from the COCO testing dataset.

<em>data_loader.py</em> is a Python file which defines the functions I used to instantiate, import, and access the COCO dataset.

<em>model.py</em> is a Python file that contains the architecture of both my CNN and my RNN networks.  

<em>vocabulary.py</em> is a Python file that defines the Vocabulary Class, including the methods I use to create the vocabulary file which the RNN architecture used to create its Embedded Word Vector.  The Embedded Word Vector of the RNN became the input which the RNN used to classify the words in the vocabulary most likely to comprise an image's caption.  
