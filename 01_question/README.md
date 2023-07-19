# Question 01 - detecting defects in given images

## Problem statement
> Write an algorithm that will detect the defect in the images given.
> Expecting a general algorithm that works for different diameters, transaltions during image acquisition, etc.

## Input images
* The given sample input images are in the [input_images](https://github.com/kdineshchitra/submission_dhvani_hackathon_computer_vision/tree/master/01_question/input_images) folder.
  
## Solution
* The [Q1-defect_detection.py](https://github.com/kdineshchitra/submission_dhvani_hackathon_computer_vision/blob/master/01_question/Q1-defect_detection.py) holds the algorithm written to detect the defects present.
* The results for the given sample images are in the [output_images](https://github.com/kdineshchitra/submission_dhvani_hackathon_computer_vision/tree/master/01_question/output_images) folder.

### Explanation
* The algorithm presented is based on image contours and convexity defective points.
* A few assumptions are made based on the given good and defective sample images.
* The major assumptions are,
  1. The input image will contain a single disk.
  2. The disk will be in black with white background.
* The defects are detected, if there are any, using the contours of disk's inner and outer ring.
* From the detected contours, a proper ring (circle-like) contour is drived.
* The differences between the masks of detected contour and drived contour are anlyzed to detect and localize the flash and cut defects.
* The algorithm ignores very small and insignificant defects, like defects with defective area of 0.01% of respective ring contour area.

### Environment setup
* Requires `Python 3.9+`
* To install the required python libraries run:
* `python -m pip install -r requirements.txt`

### Run script
* To detect:
* `python Q1-defect_detection.py sample_image.jpg`
* If a defect is detected, an output image with a defect label will be generated.
