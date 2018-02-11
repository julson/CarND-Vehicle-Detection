# Vehicle Detection

### Goals

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Apply spatial binning and color histogram to increase the number of trained features
* Implement a sub-sampling technique to obtain HOG features and use the trained classifier to search for vehicles
* Run the pipeline on a video stream (test_video.mp4 and project_video.mp4)
* Estimate a bounding box for vehicles detected

[//]: # (Image References)
[vehicle]: ./test_images/vehicle.png
[vehicle_hog]: ./output_images/vehicle_hog.jpg
[non_vehicle]: ./test_images/non-vehicle.png
[non_vehicle_hog]: ./output_images/non_vehicle_hog.jpg
[detections]: ./output_images/detections5.jpg
[heatmap]: ./output_images/heatmap10.jpg
[labeled]: ./output_images/labeled10.jpg

### Histogram of Oriented Gradients (HOG)

The `get_hog_features()` function handles the extraction of hog features and accepts orientation, pixels per cell and cells per block as parameters to determine the number of features returned.

These are what I got after HOG was applied to a single channel from one vehicle image:
![alt text][vehicle]
![alt text][vehicle_hog]

and a non-vehicle image:
![alt text][non_vehicle]
![alt text][non_vehicle_hog]

I also incorporated spatial binning and the color histograms of the training images to increase the number of feature vectors that the classifier can learn from. All 3 feature extraction methods are combined in the `extract_features()` function.

As far as the final configuration, I've settled on using 13 orientations with a spatial size of (64, 64) and 64 histogram bins, which produced a larger feature vector and seemingly less false positives. I tried using different color spaces like RGB and HSV. HSV produced a slightly higher test accuracy (~99%), but for some reason, produces a lot more false positives based on my tests.

I extracted features from 2 separate training sets for vehicle and non-vehicle images, which are then labeled 1 and 0 respectively. These are then combined, scaled and randomized with 20% of the features being used as the test set. The training set was then fed to a Linear SVM classifier (`LinearSVC()`) which produced a feature vector length of 20124 and a test accuracy of 0.9857.

### Sliding Window Search

I used an efficient sliding window search technique to search for cars in the image. Instead of running a window across the length and height of the entire image and extracting the hog features every single time, I ran HOG on a certain area in the image that's considered a lane (typically 400 <= y <= 656) and just gathered sub-samples of those hog features with the intended window size.

This is implemented in the `find_cars()` function, which also includes converting the image into the YCrCb format and running spatial binning and color histogram alongside HOG feature extraction.

Applying `find_cars()` 3 times in different scales (1.0, 1.5, 2.0) yields something like this:
![alt text][detections]

### Video Implementation

My final pipeline is set to run the `find_cars()` function in 3 scales (1.0, 1.5, 2.0) and combining all the detections. Inevitably, there will be some false positives that would sneak in (especially in shadowed areas and the median barrier). This is remedied somewhat by transferring all the detections in a heatmap and applying a threshold. Previous heatmaps are also combined to strengthen areas with cars in them and makes it easy to take out one-off false positives.

The `scipy.ndimage.measurements.label()` function was then used to create a final bounding box on areas on the heatmap that survived the thresholding operation.

![alt text][heatmap]
![alt text][labeled]

### Discussion

The main problem that I encountered was creating a reasonable false positive elimination process that still preserved positive car detections. The heatmap history approach helps a lot in removing weak false positives (typically shadows on the ground), but was pretty much unhelpful for strong false positives (like shadows on the median barrier) that typically manifest as multiple detections clustered in an area. Sometimes the classifier performs 'too well' and even detects cars from the opposing lane.

I tweaked some parameters like changing color spaces, scales and increasing the number of orientations, but the results were marginal at best. Smaller scales tend to produce more vehicle detections at the cost of more false positives, while larger scales produced the opposite, which made setting a threshold value more difficult. Creating non-vehicle training images from problematic areas in the video helped in reducing the detections from the shadows, but did not help in stopping detections of cars across the barrier (I guess the classifier was doing a good job in this case? haha).

A bigger training set (like the ones supplied by CrowdAI) for both vehicle and non-vehicle sets would certainly help in producing more accurate vehicle detections. One thing I can also try is to not rely on a single color space, but to pick out channels from different color spaces (like H from HSV) that may produce better results from HOG.
