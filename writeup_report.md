#**Traffic Sign Recognition** 

##Writeup Report

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visualization.png "Visualization"
[image2]: ./images/preprocess.png "Preprocess"
[image3]: ./images/pred_new_images.png "New images prediction"
[image4]: ./images/topk.png "New images Top K"


###Data Set Summary & Exploration

####1. A basic summary of the data set.

The code for this step is contained in the [2] code cell of the IPython notebook.

I used the python built-in function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

The code for this step is contained in the [3] code cell of the IPython notebook.  

Here is an exploratory visualization of the data set.
For each class, it plotted out the number of training samples together with
a training sample image.

![alt text][image1]

###Design and Test a Model Architecture

####1. Preprocessing of image data.

The code for this step is contained in the [4] code cell of the IPython notebook.

As I browsed some images through the dataset, I found that quite a few of those images
were dim. So I decided to improve the contrast. 

First step of the code converted a image from RGB channels to HSV channels,
secondly applied [CLAHE](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
method on the luminance channels, and then converted back to RGB channels.

But CLAHE would enhance the noise all the same, thus I used a [bilateralFilter](http://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed)
to smooth images.

At last I normalized all the color channels into range from 0 to 1.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image2]

####3. Model Architecture.

The code for my final model is located in the [7] cell of the ipython notebook. 

My final model which used LeNet structure, consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x24	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x24 					|
| Fully connected		| 864 to 256 hidden layer     					|
| RELU					|												|
| Fully connected		| 256 to 128 hidden layer     					|
| RELU					|												|
| Output				| 128 to number of class      					|


####4. Model Training.
The code for training the model is located in the [9-11] cell of the ipython notebook. 

To train the model, I used:

* AdamOptimizer
* batch size of 128
* epochs of 100

####5. Solution Approach
The code for calculating the accuracy of the model is located in the [12] cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.999%
* validation set accuracy of 96.2%
* test set accuracy of 94.7%

I chose LeNet architecture at first. My LeNet consisted two pairs of convolution and max poolng layers,
one for 5x5 patch size and another for 3x3, then followed by two hidden layers,
connecting to a output layer at the end.

I thought LeNet is a small but efficient enough for this traffic sign classification
task.
Those two convolution layers allowed the network to scan over images 
and to learn some texture pattens with different size of pattens by shared weights.
A little bit of Dropout(0.95) was added to prevent over-fitting.

LeNet had been proved its ability of recognizing hand written digits, 
traffic signs are also symbols not much more complicated than digits,
so I believed LeNet could handle this task.

I used a learning rate of 0.0008 and 100 epochs to train the network,
feeding the Adamoptimizer(Ada delta gradient descending, providing 
learning rate decay functionalities) with a batch size of 128 while training.
That should be enough for the model to return a pleasant result after training.
The training grew quickly above 99%, and then validation accuracy raised to 95%.
The model got 94% test accuracy very close to validation accuracy, which has met
my expectation.

I also tried a Inception structure, but it was too slow to train since it had a lot
more parameters than a simple LeNet model. It took too long to train even on a
AWS GPU server...

###Test a Model on New Images

####1. Acquiring New Images

Here are ten German traffic signs that I found on the web:
![1](http://bicyclegermany.com/Images/Laws/Stop%20sign.jpg)
![2](http://media.gettyimages.com/photos/german-traffic-signs-picture-id459381059)
![3](https://is.alicdn.com/img/pb/312/820/215/1215820312_482.jpg)
![4](http://media.gettyimages.com/photos/german-traffic-signs-picture-id459381063)
![5](https://cdn.pixabay.com/photo/2016/06/08/01/41/traffic-sign-1443060__480.jpg)
![6](http://bicyclegermany.com/Images/Laws/100_1607.jpg)
![7](https://francetaste.files.wordpress.com/2016/03/speed-reminder.jpg)
![8](https://thumb1.shutterstock.com/display_pic_with_logo/3869111/362901644/stock-photo-german-speed-limit-sign-km-h-against-blue-sky-362901644.jpg)
![9](http://media.gettyimages.com/photos/german-traffic-signs-picture-id459381091)
![10](http://media.gettyimages.com/photos/german-traffic-signs-picture-id459381023)

The fourth image might be difficult to classify because there was fewer training samples
of its class and it had a  noisy background.

####2. Performance on New Images

The code for making predictions on my final model is located in the [15-16] cell of the Ipython notebook.

Here are the results of the prediction:

![alt text][image3]

The model was able to correctly guess 7 of the 10 traffic signs,
which gives an accuracy of 70%.
It seemed that the model could not work very well in the reality. Sadly it was less
able to read numbers...

####3. Softmax Probabilities
The code for making predictions on my final model is located in the [17] cell of the Ipython notebook.

![alt text][image4]

For all images, the model was rather sure of all its first choices,
with probabilities over 99%. However, it gave wrong classification results
for the 7th, 8th and the 9th images.

For the seventh image, the model returned “No passing” as result, which actually
was “Speed limit(50km/h)”.

For the eighth image, the model returned “Speed limit(60km/h)” as result, which actually
was “Speed limit(70km/h)”.

For the ninth image, the model returned “No passing” as result, which actually
was “Speed limit(60km/h)”.
