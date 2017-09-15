# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

[//]: # (Image References)

[image0]: ./examples/test_aug.png "Image preprocessed"
[image1]: ./examples/test_clr.png "Original"
[image2]: ./examples/test_clr_flip.png "Flipped Version"
[image3]: ./examples/center_2016_12_01_13_32_43_154.jpg "Center Image"
[image4]: ./examples/left_2016_12_01_13_32_43_154.jpg "Left Image"
[image5]: ./examples/right_2016_12_01_13_32_43_154.jpg "Right Image"

## Files Submitted & Code Quality

### 1. Required Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

### 2. Code Functionality
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```
python drive.py model.h5
```

### 3. Usebility & Readability

The model.py file contains the code for training and saving the convolution neural network with a Keras generator. The file shows the pipeline I used for training and validating the model. It contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. Model Architecture

The model architecture was derived from the Nvidia convolutional neural network for "End to End Self-Driving Cars" [source](https://arxiv.org/pdf/1604.07316v1.pdf). 
My network consists of 3 convolutinal layers with stride 2 each, a quadratic kernel size of 5 and depth between 24 and 48. ELUs were used as nonlinearities, Dropout with 50% drop probabilty and Keras lamda layer for data normalization.
The main difference to the original Nvidia network lies in using zeropadding in the first two convolutional layers to reduce the downscaling of my data and using one less fully connected layer due to a much smaller feature map before the first fully connected layer. 
Changes were made to handle small image size of 80x20x3 (WxHxC). Smaller inputs were choosen to reduce training time. The final pipeline, including data handling, trained the network on my MacBook Pro 2011 in about 4 minutes.

### 2. Overfitting

As in the original Nvidia network, a dropout layer with 50% drop probability was used on the last convolutional layer to reduce overfitting the data. Further, the model was trained and validated, using 10% of the data for validation. Testing was performed by running the network on the autonomous mode of the simulator.
A form of early stopping was used to find the best working model on the test track in autonomous mode. Therefore, one model was saved per training epoch and each models performance was measured on the test track by visual inspection. It was found, that whether the training nor the validation loss were reliable inidcators for the models performance. Models in epoch 7 to 9 were able to drive around the track, while sooner or later model snapshots were not.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### 4. Appropriate training data

For training data, only the provided data set was used due to issues with creating new data with the simulator. When driving with the mouse I was not able to handle to car as desired. The new data would only produce worse results than the model trainined on the provided data.
Minor data augmentation was used to enlarge the data presented to the network with Keras generator:
* Additional to the center image the left and right camera images were used to provide data where the car is drifting off to the left or right side of the road. A correction factor of +/-0.2 is added to the steering angle, so the car tends to drive back to the middle of the road:

![alt text][image3] ![alt text][image4] ![alt text][image5]

* The image was flipped by 50% chance and the steering angle inverted to balance the possibility of steering left or right:

![alt text][image1] ![alt text][image2]

* Data was preprosessed to cut off unnecessary image information. Therefore, the lower 20 pixels (hood of the car) and the upper 60 pixels (sky and trees) of the image were cropped and then downscaled with a factor of 4 from 320x80x3 to 80x20x3.

![alt text][image0]

## Model Architecture and Training Strategy

### 1. Solution Design Approach

* My first thoughts were, that I want to be able to train a small network on my slow CPU. Therefore, I decided from the beginning to use small input images. Later I found the CommaAi network which uses strides 8 to reduce feature map sizes early in the network, but I do not have time to test this.
* At first I worked without a generator but used data the discribed data augmentation approach. The network was not able to drive around the track.
* I adapted my code to work with a generator with the same data augmentation approach. I trained the network over 15 epochs with around 7000 samples. The network was still not able to drive around the track. As before the network had the tendency to drive staight in curves, because the steering angle of 0 is overrepresented in the data set.
* I used early stopping and evaluated one network after each epoch on the test track. My network was able to drive around the track, when I used the snapshots after 7 to 9 epochs. These models were not as overfitted to the data and therefore were able to generalize better on unseen situations on the test track.
* I would like to try other data augmentation strategies like random shadows, brightness variations or translations in x and y direction and also create more data with driving the simluator by myself with a controller in the future, but for the first track the choosen data augmentation and the provided data set seem to be suffient.

### 2. Final Model Architecture

The final model architecture is a convolutional neural network derived from the Nvidia network and some adaptations due to smaller input size:

| Layer         		| Layer Parameter                |     Output Dimension     	        					| 
|:---------------------:|:---------------------:|:---------------------------------------------:|
| Input         		| -     | 80x20x3 RGB (0 to 255) 							| 
| Normalization   	    | -     | 80x20x3 RGB (-0.5 to 0.5)	|
| Convolution           | 5x5x24, zeropadding| 40x10x24											|
| Convolution           | 5x5x36, zeropadding| 20x5x36 			|
| Convolution           | 5x5x48| 1x1x48      									|
| Dropout               | 50%   | 1x1x48        									|
| Flatten 				| -     | 48        									|
| Fully connected		| 100   | 100											|
| Fully connected       | 50    | 50										|
| Fully connected 	    | 10    | 10				|
| Output				| 1     | 1 Steering angle										|

### 3. Creation of the Training Set & Training Process

Training data set was created by splitting the provided data set randomly into 90% training and 10% validation images each epoch. The training data was augmented by using all three camaera images (center, left, right) and change the steering angles accordingly. Furthermore, the images were flipped with a chance of 50% and the steering angles inverted while training. Example images are provided above.
