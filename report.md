# Introduction


# Test Platform Specifications


# Experiment Design
There are 3 main ways to enhance OpenALPR's results:

1. Train it with tagged number plates. Number plates regions are selected in the image by an operator, and the number plate and region are saved in the system. This is then fed into OpenALPR.
1. Train it using the country's fonts. A special grid representations of the fonts used in the country's number plates is made. This includes all the characters used in the font, as well as each character's distorted represenations (this means that cameras viewing the plate at an angle work too). In order to achieve this, the fonts used in every state and territory need to be collected. In Australia, these are managed by a company called Licensys.
1. Caliberate each camera used, so that the images can be rotated (in 3D) reducing the distortion of the letters and numbers on the plate. The rotation matrix for the camera is saved, and applied to each image from that camera.

We need to test each of these enhancements separately, and together.

# Results

## Completely untrained and uncaliberated system

## Cameras Caliberated on untrained system

## System trained with half the number plate images

## System trained with half the number plate images and available fonts


# Conclusion
