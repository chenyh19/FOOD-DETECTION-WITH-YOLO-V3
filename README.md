# FOOD DETECTION WITH YOLO V3
Hello world, what's up? Today I am coming with another repository about deep learning. This repository aims to use YOLO V3 to detect some food (salad and fruit) on images.

## INTRODUCTION
In restaurants, it could be interesting to automate payment instead of waiting in line for hours to be checked out by a cashier. 
To do so, we could use a camera that will automatically detect the content of a customer tray and calculate the cost of their order.
Here, we are focusing on the detection part and trying to detect the "salad and fruit" containers and classify them according to the
size of the container. "Big vrac" corresponds to the big containers and "small vrac", to the samll containers.
Here is a picture of the two types of "salad and fruit" container we want to detect.
<p float="left">
    <img src="Images/image1.PNG" width="425"/> 
</p>

## SOME INFO ABOUT THE PROJECT
I worked on 1000 images. Each image is associated to one or more bounding boxes [xmin, ymin, width, height] and
the corresponding classification labels(small_vrac or big_vrac). I evaluated the model performances by computing the mAP(mean Average Precision). 
Our model ended up with a mAP equals to 0.33 on the test set.

As you can see on the next images, some predictions are better than the corresponding ground truths.
<p float="left">
    <img src="Images/image2.PNG" width="425"/>
</p>
<p float="left">
    <img src="Images/image3.PNG" width="425"/>
</p>

But it's not always the case. In order to better the detection and classification results, it could be nice to use data augmentation.

## UNDERSTAND THE FILES AND FOLDERS

-yolo3 : This folder contains :
    -utils.py, a file with some miscellaneous utility functions.
    -Model.py, a file with the functions related to the CNN model construction and training.

-model_weights : is the folder where we save the weights during training.The most important file of this folder is :
    -trained_weights_final.h5, a file containing the final weights of our trained model. You can load these weights to detect salad and  fruit images on new images.

-images : is the folder containing some of the images I used to train and test my model.

-dataset2020216133642 : is the folder where I saved the test and train sets I used. It contains :
   -test.npy, my test set
   -train.npy, my training set.

-yolo_weights.h5 : corresponds to the pretrained weights I used to initialize my model weights

-Training_phase.ipynb : is the notebook I used to train my model(I did the training on google colab). It allows to see the code more clearly.

-Detect.ipynb : is the notebook I used to detect the objects on my images. You can use it to visualize the results of the detection I made on some images and apply the code on your own images. The detect_and_classify_vrac(im_path)can be found there.
