## Writeup: Advanced Lane Detection.

### Dependencies:
1. Python 3.6
2. OpenCV Library
3. Pickle module
4. Matplotlib 
5. Numpy
6. Glob

### Folders in the directory:
	1. camera_cal: It contains images taken by my personal GoPro camera.
	2. camera_cal_udacity: It contains images provided for the project.
	3. output_images: It contains undistorted images of my camera and the warped lane images. 
	4. output_images_udacity: It contains undistorted images and warped lane images for the project.
	5. test_images: It contains various lane images taken by my camera.
	6. test_images_udacity: It contains lane images provided for this project.
	7. writeup_images: It contains images that are used for the README file.
	
### Files in the directory:
- **Pickle files:**
	1. *imgpoints.pickle*: Pickle file containing image points for calibrating personal camera.
	2. *imgpoints_udacity.pickle*: Pickle file containing image points for calibrating camera on udacity self driving car.
	3. *objpoints.pickle*: Pickle file containing object points for calibrating personal camera.
	4. *objpoints_udacity.pickle*: Pickle file containing object points for calibrating camera on udacity self driving car.
- **Video files:**
	5. *project_video.mp4*: Video provided for this project.
	6. *Augumented_video.avi*: Output video of the pipeline with marked lanes.
- **Jupyter Notebooks:**
	7. *Advanced Lane Finding-Camera Calibration.ipynb*: Jupyter notebook for personal camera calibration and storing the undistorted images.
	8. *Udacity-Camera Calibration.ipynb*: Jupyter notebook for udacity camera calibration and storing the undistorted images.
	9. *Pipeline(Image)*: Jupyter notebook for finding lanes in the image and marking them.
	10. *Pipeline(Video)*: Jupyter notebook for finding lanes in the video and saving the output video.
- **Markdown File:**
	11. writeup.md: Containing writeup of the project.
	
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/2.jpg "Distorted"
[image1]: ./output_images/2_undistorted.jpg "Undistorted"
[image2]: ./camera_cal_udacity/calibration1.jpg "Distorted"
[image3]: ./output_images_udacity/1_undistorted.jpg "Undistorted"

[image4]: ./test_images_udacity/test6.jpg "Road"
[image5]: ./writeup_images/test6_undistorted.jpg "Road Transformed"

[image6]: ./writeup_images/HLS.jpg "HLS"
[image7]: ./writeup_images/hls_binary.jpg "Binary HLS"
[image8]: ./writeup_images/thresholding.jpg "Thresholding"
[image9]: ./writeup_images/combine.jpg "Combined thresholding"
[image10]: ./writeup_images/hls_combine.jpg "Combined HLS and thresholding"

[image11]: ./writeup_images/birds_eye.jpg "Birds_eye View"

[image12]: ./writeup_images/download.jpg "Binary Top View"
[image13]: ./writeup_images/windows.jpg "Windows in the image"
[image14]: ./writeup_images/histogram.jpg "Histogram"
[image15]: ./writeup_images/1st_window.jpg "First Window"
[image16]: ./writeup_images/roi.jpg "Finding Lane Pixels"
[image17]: ./writeup_images/lane_finding.jpg "Lane Lines"

[image18]: ./writeup_images/radius.jpg "Calculating radius of curvature"

[image19]: ./writeup_images/augumented.jpg "Augumented Image"

[video1]: ./Augumented_video.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### 1. Camera Calibration
### 2. Pipeline(Image)
1. Example of distortion corrected image
2. Methods to create Binary image
3. Perspective Transform of the image
4. Identifing lane-line pixels and fit their positions with a polynomial
5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.
6. Annotated result on an image.
### 3. Pipeline(Video)
---


## 1. Camera Calibration

The code for obtaining undistorted image is located in the IPython notebook "**Advanced Lane Finding-Camera Calibration.ipynb**" and "**Udacity-Camerea Calibration.ipynb**". Following are the distorted and the undistorted image obtained using OpenCV library.

![alt text][image0]
![alt text][image1]


- Due to the lenses used in the camera, we can't get a perfect picture from a camera. Due to radial and tangential distortion, we will get a distorted image which will result in curving the straight edges. To counter this effect we have to undistort the image using mathematical transformations. In above mentioned notebooks, code to get undistorted image is explained. 

- I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

- I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]
![alt text][image3]

The `objpoints` and the `imgpoints` are stored using Pickle module. Following are the pickle files for my personal camera ***imgpoints.pickle*** and ***objpoints.pickle*** while ***imgpoints_udacity.pickle*** and ***objpoints_udacity.pickle*** are for the udacity camera.

## 2. Pipeline (Single Image)
- Code for this pipeline is located in the Pipeline(Image).ipynb
- The code is divided into 3 parts: 
 1. Function definition (Code cell 1 to 20)
 	 - All the functions which are used in the pipeline are defined.
 2. Pipeline (Code cell 21)
 	 - Importing image, passing it through all the functions and then displaying lane marked image. 
 3. Display the process. (Code cell 22 to 34)
 	 - Step-by-step operations on the image are displayed.

#### 1. Provide an example of a distortion-corrected image.
Following is the distorted test image.
![alt text][image4]

- To undistort it, firstly I have imported pickle files(saved in camera calibration code) using `load_pickle()` function(*2nd code cell in the Pipeline(Image).ipynb*). Later I passed **distorted image**, **objpoints** and **imgpoints** as input parameters to `cal_undistort()` function. 
- This function will return undistorted image. 
- It uses `cv2.calibrateCamera` and `cv2.undistort` functions to return undistorted image. 
- Following is the undistorted image obtained using `cal_undistort()` function(*3rd code cell in the Pipeline(Image).ipynb*).

![alt text][image5]


----

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

- I used a combination of color and gradient thresholds to generate a binary image (*thresholding steps at lines 25 through 45 in `main()` of Pipeline(Image).ipynb*).
- **Color Thresholding**: Since RGB channels of the image are sensitive towards lighting conditions, I have used HLS color space for thresholding.
- `HLS()` function gives Hue, Saturation and Lighting channels of the image.
- `HLS()` function is located in *6th code cell in the Pipeline(Image).ipynb*
- Function Intakes: undistorted image 
- Function Returns H, L, S channels of the image. 
- Following are the grayscale, H, L, and S channels of the undistorted image.

![alt text][image6]

- I have then converted these channels to binary image by using thresholding.
- `HLS_binary()` function was used to obtain these results.
- `HLS_binary()` function is located in *7th code cell in the Pipeline(Image).ipynb*
- Threshold value: minimum = **150** and maximum = **255**. 
- Function Intakes: `gray, H, L, S`
- Function Returns: `binary_gray, binary_H, binary_L, binary_S` 
- Following are the binary result images:

![alt text][image7]

- **Gradient Thresholding**: Along witht the color thresholding, I have used Sobel x-gradient, Sobel y-gradient, magnitude thresholding and directional thresholding.
- Functions used: `abs_sobel_thresh()`(*8th code cell*), `mag_thresh()`(*9th code cell*) and `dir_threshold()`(*10th code cell*)
- Function intakes: `undistorted` image.
- Function returns: `gradx/grady`, `mag_binary`, `dir_binary` respt.
- Kernal size for all 3 functions: 3
- Threshold values (min, max):
	1. For `abs_sobel_thresh()`: (30, 130)
	2. For `mag_thresh()`:       (70, 255)
	3. For `dir_threshold()`:    (0.7, 1.3)

- Following are the outputs of these functions:

![alt text][image8]


- Later, I have done bitwise AND operations on `gradx & grady` and `mag_binary & dir_binary`.
- Later, bitwise OR operation on the result of above two.
- This gave me the pixels only which are overlapping in those images.
- Function used: `combine_sobel()`(*11th code cell*)
- Function intakes: `gradx`, `grady`, `mag_binary` and `dir_binary`
- Function returns: `combined`
- Following is the result of `combine_sobel()`

![alt text][image9]

- I performed bitwise OR on the binary images obtained from color thresholding and gradient thresholding.
- I choose binary Saturation channel(`binary_S`) since it performs good with different lighting conditions.
- Function used: `combined_HLS()`(*12th code cell*)
- Function intakes: `dir_threshold`, `combined`, `binary_S`
- Function return: `new_combined`
- Following is the result of `combined_HLS()` function.

![alt text][image10]

---

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

- The code for my perspective transform includes a function called `warp()`, which is in the 13th code cell of the IPython notebook.  
- The `warp()` function takes as inputs an image (`img`).
- It returns `warped`, `Minv`, `warped_lanes`.
- `warped` is the Perspective Transform of the (`img`) passed through the function.
- `Minv` is the Inverse Persecptive Transform matrix.
- `warped_lanes` is the Perspective Transform of the `masked_edges`
- I chose the hardcode the source and destination points in the following manner:


    	src = np.float32(
	    	[[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
	    	[((img_size[0] / 6) - 10), img_size[1]],
	    	[(img_size[0] * 5 / 6) + 60, img_size[1]],
	    	[(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    	dst = np.float32(
	    	[[(img_size[0] / 4), 0],
	    	[(img_size[0] / 4), img_size[1]],
	    	[(img_size[0] * 3 / 4), img_size[1]],
	    	[(img_size[0] * 3 / 4), 0]])


This resulted in the following source and destination points:

- | Source        | Destination   | 
- |---------------|---------------| 
- | 585, 460      | 320, 0        | 
- | 203, 720      | 320, 720      |
- | 1127, 720     | 960, 720      |
- | 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image11]

---

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Below is the image obtained as a output of `wrap()` function. Finding out lane pixels is easy in this image compared to the imported image so I have foundout the lane pixels in this image. Code for lane pixel finding is in the *14th code cell of the IPython notebook*. It contains a function `lane_finding()` which intakes binary birds-eye view image(image displayed below).
![alt text][image12]

- Above image shows the lane pixels in white color. 
- To findout lane pixel coordinates, we will divide this image in number of windows(here 5).
- The window height was decided based on image size. Image height(720) was divided into 5 equal parts thus making the height of 144 for each window.
- Code for lane finding is in `lane_finding()` *14th code cell of the IPython notebook*
- | Window Number | `win_y_high` | `win_y_low`|
- |:-------------:|:------------:|:----------:|
- |      1        |     720		 |    576     |
- |      2        |     576      |    432     |
- |      3        |     432      |    288     |
- |      4        |     288      |    144     |
- |      5        |     144      |     0      |
- Below image shows 5 windows on `warped` image.

![alt text][image13]
**Histogram:** To avoid searching blindly in these windows, I took a histogram of the image and checked for the peak values to findout the lanes.

![alt text][image14]

- To findout the lanes, I moved from window 1 to 5 sequentially. I divided the image vertically into 2 parts  i.e. from 0-640 and 641-1280 pixel coordinates.
- The pixels in first half belongs to left lane while the other half to right lane.   
- I kept appending all the nonzero lane pixels coordinates in `left_lane_inds` and `right_lane_inds`.

![alt text][image15]

- Using peak values from histogram, I selected a region of interest in the window.
- The region of interest was selected by adding +-100 pixels to the peak value coordinates. This way, I got 2 rectangles(purple colored) in below image where I searched for nonzero pixels.
- The nonxero pixels are the coordinates of lane pixels.
- All these pixles are appended to the `left_lane_inds` for left lane and `right_lane_inds` for right lane.
- Same thing I have repeated for all other windows and store the pixel coordinates in `left_lane_inds` and `right_lane_inds`.

![alt text][image16]

By doing above mentioned procedure for all windows, I got x and y coordinates of all the nonzero lane pixels. Fitting second order polynomial to left and right pixel coordinates, I got a curve on each side which represents lane lines. Below image shows those lines.

![alt text][image17]


---

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

**Radius:**

- Below is an image showing two curve lane with their respective second order of polynomial equation.
- The radius is calculated using equation
R_curve​​ = ​​(1+(2Ay+B)^2)^(3/2))/(2A). Values of A, B and C I got from the polyfit function.
- `radius_pixels()`(*16th code cell of the IPython notebook*) gives radius in terms of pixels.
- `radius_m()`(*17th code cell of the IPython notebook*) gives radius in terms of meters.

![alt text][image18]

**Offset:**

1. Since camera is attached at the center of the vehicle, vehicle center is same as the image center.
2. Difference between the coordinate of the points on right and left, gives lane width.
3. Therefore, lane center = left_point + (lane_width/2)
4. Difference between vehicle center and the lane center gives the offset from center of the lane.
5. `offset_cal_pix()`(*18th code cell of the IPython notebook*) gives offset value in terms of pixels.
6. `offset_cal_met()`(*19th code cell of the IPython notebook*) gives offset value in terms of meters.

---

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

- After drawing lane curves on the image, I used Inverse Perspective Transform to draw back those lanes on the original image.
- The area covered under these lanes is painted green and the resulting image is displayed with the `Radius of Curvature` and `offset` value.
- `augumentation()` (*20th code cell of the IPython notebook*) is used to all this.
- Following is the final augumented image which clearly shows lanes.

![alt text][image19]

---

### Pipeline (Video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Same pipeline is applied to number of images i.e. Video and the following result is saved.
Here's a [link to my video result](./Augumented_video.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Issues:**

- Selecting proper `src` and `dst` points.
- Car affecting lane detection: Sometimes, it might detect car pixels as lane pixels.
- Not smooth: Lane finding is done for each and every frame. If we use past few frame's lanes coordinates to take an average and then decide 

**Fail:**

- Curvy road
- Sudden lighting variation
- Lanes too much faded


**Improvements:**

- Smoothing 