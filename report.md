# Introduction


# Test Platform Specifications


# Experiment Design
There are 3 main ways to enhance OpenALPR's results:

1. Train it with tagged number plates. Number plates regions are selected in the image by an operator, and the number plate and region are saved in the system. This is then fed into OpenALPR.
2. Train it using the country's fonts. A special grid representations of the fonts used in the country's number plates is made. This includes all the characters used in the font, as well as each character's distorted represenations (this means that cameras viewing the plate at an angle work too). In order to achieve this, the fonts used in every state and territory need to be collected. In Australia, these are managed by a company called Licensys.
3. Caliberate each camera used, so that the images can be rotated (in 3D) reducing the distortion of the letters and numbers on the plate. The rotation matrix for the camera is saved, and applied to each image from that camera.

We need to test each of these enhancements separately, and together.

# Results

## Completely untrained and uncaliberated system

Accuracy: 4.73%

## Cameras Caliberated on untrained system

Accuracy: 10.17%

For each location ID and camera ID, a separate rotation matrix config file has been created and each image is calibrated based on the rotation matrix, before the number plate recognition exercise is carried out.

Some anecdotal observations from the calibration exercise (with a given set of images) are as follows:

1. OpenALPR has a higher accuracy even without calibration when attempting to recognise vehicle number plates from a closer range (distance between the camera and the vehicle plate). However, the images where the distance is greater, OpenALPR performs poorly even after calibration.

2. A higher accuracy was observed in images at day as opposed to at night.

3. A lower accuracy has been observed with images which have a specific plate style (375 x 90 mm with a white background and a black font) in comparison to others.

4. Accuracy improves just with by zooming into the image (and not over-zooming).

## System trained with half the number plate images

## System trained with half the number plate images and available fonts


# Conclusion
