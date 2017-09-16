
# **Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidYellowCurve.jpg

[shortcome1]: ./shortcome/fail_to_get_uncontinus_line.png

[shortcome2]: ./shortcome/strong_light_case.png

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline is encapsulated in function trans_img, which use an source image as parameter and return source image with lines drawn on it.
The function trans_img consists of 5 steps.
1. Convert the source image to grayscale;
2. Apply Gaussian noisy kernel with kernel size = 5;
3. Canny edge detection with threshold 100 and 200;
4. Extract region of interested;
5. Apply Hough transform with rho = 1, theta=pi/180, threshold=50, min_line_len=20, max_line_gap=100;
6. Weighted the source image with lines got in step 5.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by:
1. Calculate the slope of each line and drop those lines that are too even;
2. Divide these lines into two sets by check whether its slope is positive or not. Negative sets represent left line and positive sets represent right line;
3. Use linear regression method in numpy to get extimated lines for each set;
4. Find the cross points of estimated lines with bottom edge of the image;
5. Draw each line with one endpoint is cross point and the other is the furthest point from bottom.

![result][image1]



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be no line detected when sometimes line on roads are not continuous and quite short for each segment.

![result][shortcome1]

Another shortcoming could be no line detected when the sunlight is too strong and makes it hard to distinguish lane line from road. Besides shadows can be detected as lines sometimes.

![result][shortcome2]

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to extract yellow lines with color info not only gradient.

Another potential improvement could be to use the lines of last several frames as prior knowledge, since there are some relationship between neighbor frames in a vedio and sometimes our pipe line cannot find out lines of lane because of light, weather etc. Parameters can be changed like interested region and canny edge thresholds according to prior knowledge.
