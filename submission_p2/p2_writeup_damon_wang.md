#**Traffic Sign Recognition** 

##Writeup

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

[imagev1]: ./my_results/train_hist.png "Bar chart of Training data"
[imagev2]: ./my_results/valid_hist.png "Bar chart of Validating data"
[imagev3]: ./my_results/test_hist.png "Bar chart of Testing data"
[imagev4]: ./my_results/image.png "Visualization of am image"

[imagep1]: ./my_results/simple_preprocess.png "Simple preprocess of am image"
[imagep2]: ./my_results/gray_norm_preprocess.png "Grayscale of am image"

[imagew1]: ./web_img/s_caution.png "Caution"
[imagew2]: ./web_img/s_no_entrance.png "No Entry"
[imagew3]: ./web_img/s_turn_right.png "Turn Right"
[imagew4]: ./web_img/s_slippery_road.png "Slippery Road"
[imagew5]: ./web_img/s_speed_limit_60.png "Speek Limit 60"

[imagef1]: ./my_results/feature_map.png "Feature map"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jingramgithub/udacity_self_driving_car/blob/master/submission_p2/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the basic Python methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
![Am image from training data set][imagev4]

There are bar charts showing how the training, validating, testing data distributed. It shows distribution of each label is similar in different set.

![Training data][imagev1]![Validating data][imagev2]![Testing data][imagev3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I just used a simple way as (pixel - 128)/ 128. But the final Validation Accuracy only reach 0.905 after 25 epochs. Here is the preprocessed result of a training data. 
![Simple preprocess of image][imagep1]

I guess there are some color noise on images after such simple ways. Maybe it's better to use grayscale for images first.

![Grayscale preprocess of image][imagep2]

However when apply trained model to images I found from website, it's less accurate than using color images. So I finally removed these operations which transform images to grayscale. Instead I added more layers and tuned other parameters to make the training more accurate with validating and training sets.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                             | 
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Max pooling 2x2       | 1x1 stride,  outputs 27x27x6                  |
| Convolution 7x7       | 1x1 stride, valid padding, outputs 21x21x12   |
| RELU                  |                                               |
| Max pooling 2x2       | 1x1 stride,  outputs 20x20x12                 |
| Convolution 9x9       | 1x1 stride, valid padding, outputs 12x12x16   |
| RELU                  |                                               |
| Max pooling 2x2       | 2x2 stride,  outputs 6x6x16                   |
| Flatten               |                                               |
| Fully connected       | outputs 120                                   |
| RELU                  |                                               |
| Fully connected       | outputs 84                                    |
| RELU                  |                                               |
| Fully connected       | outputs 43                                    |
| Softmax               | etc.                                          |
|                       |                                               |
|                       |                                               |
 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with batch size is 128, number of epochs is 3 and learning rate is 0.001.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.987
* validation set accuracy of 0.944
* test set accuracy of 0.918

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I chose sigmoid as activation functions for all layers and use grayscale to preprocess images. Because sigmoid is more familar to me. Using grayscale is because I think it can make things simpler. I used the same structure of example of LeNet in previous classes.

* What were some problems with the initial architecture?

Using sigmoid needs more epoches to achieve a better accuracy.
Using grayscale results more failure on images I got from website.
Training acccuracy is not good enough and need many epoches to reach 0.93.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I changed sigmoid to Relu and added one more layer, it then becomes much faster to reach 0.93.
I removed steps which apply grayscale transform to images. Because it result in a bad recognition with images from web. I guess there are color infos which is helpful.

* Which parameters were tuned? How were they adjusted and why?

I changed filter size in conv layer to bigger value. It can get more accurate result in training set and validating set.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Keep images to have depth of 3 can give more information to our model. So it's more robust when facing with signs in other environment like standard signs from wikipedia.
Add one more layer and set larger filter size can make the result much much better.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web(https://en.wikipedia.org/wiki/Road_signs_in_Germany):

![alt text][imagew1] 
![alt text][imagew2] 
![alt text][imagew3] 
![alt text][imagew4] 
![alt text][imagew5]

The last image might be difficult to classify because speed limit 60 is quite similar to speed limit 50. But it is accurate. However turn right sign misrecognized into speed limit 20 is quite strange.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| caution               | caution                                       | 
| No Entry              | No Entry                                      |
| Turn Right            | Speed limit (20km/h)                          |
| Slippery Road         | Slippery Road                                 |
| Speed limit 60 km/h   | Speed limit 60 km/h                           |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all these image I got from website, the model is relatively sure, most of the highest probability are almost 0.9999), and the image does contain those signs. Only the turn right sign is not correct. The top 5 softmax are as follows:

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .973                  | Speed limit (20km/h)                          | 
| .025                  | Turn Right                                    | 
...

The other prediction's probability are lower than 0.001.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![Feature maps][imagef1]
 
  Mostly these edges in image are characteristics the model uses to make classification.
