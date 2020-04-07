# Introduction


# Test Platform Specifications
We are running the tests on MacbookPro laptops, using a modified version of openALPR to run on these. The version in master does not compile against opencv4 (we've put in a pull request to rectify this, but until it's merged, please clone from Sara's branch:

```
git clone https://github.com/sarafalamaki/openalpr.git
git co macos-catalina
```

You can then follow instructions on in the documentation to make and make install it:
https://github.com/openalpr/openalpr/wiki/Compilation-instructions-(OS-X)

We have used the openalpr python bindings to write scripts to experiments.



# Experiment Design
There are 3 main ways to enhance OpenALPR's results:

1. Train it with tagged number plates. Number plates regions are selected in the image by an operator, and the number plate and region are saved in the system. This is then fed into OpenALPR.
2. Train it using the country's fonts. A special grid representations of the fonts used in the country's number plates is made. This includes all the characters used in the font, as well as each character's distorted represenations (this means that cameras viewing the plate at an angle work too). In order to achieve this, the fonts used in every state and territory need to be collected. In Australia, these are managed by a company called Licensys.
3. Caliberate each camera used, so that the images can be rotated (in 3D) reducing the distortion of the letters and numbers on the plate. The rotation matrix for the camera is saved, and applied to each image from that camera.

We need to test each of these enhancements separately, and together.

Potential areas for further experiments:

Pattern matching: Currently the patterns mentioned in the au.patterns file in the runtime directory (also mentioned in the readme) are in no particular order, but will be read by OpenALPR in order of likelihood, so this is another area for further experimentation i.e. re-arranging the order of the patterns may improve accuracy. Also need to consider how to eliminate particular patterns, if need be.

# Training

There are several steps involved in training the system. To save you going through the many different repos in OpenALPRs github, we provide an overview on how we're doing the training here.

1. Copy all the image files into one directory. Lets call this directory `/tmp/pool` Before you do this, make sure each image has a unique filename. We're assuming all images have similarly formatted filenames for this exercise, as we're getting them all from Revenue NSW.

1. Run the plate tagger utility:
```
git clone https://github.com/openalpr/plate_tagger.git
cd plate_tagger
mkdir build && cd build
cmake ..
make
./openalpr_tagger
```
You will pick out numberplates for all your images, and tag them with the contents of the numberplate. This will create 1 yaml file for each jpg file in the dir.

1. Break the images into directories based on location and camera. The packages
   we've been provided contain images from 2 or 3 different cameras per
incident. You can put the images in the appropriate directories by running the
following function: 
```
cd src
mkdir ../data/pooled_dirs
mkdir /tmp/pool

python3
>>>import create_labeled_data
>>>create_labeled_data.put_in_directories("/tmp/pool", "../data/pooled_dirs", "jpg")
>>>create_labeled_data.put_in_directories("/tmp/pool", "../data/pooled_dirs", "yaml")
```  

You'll need to do this for the jpg files and the yaml files separately.

1. Run the first 2 experiments. 

1. Now you have to use your yaml files to crop the plates in order to train tesserect. 

```
git clone https://github.com/sarafalamaki/train-detector.git
```
..and follow the instructions. I've raised a PR, but it hasn't been merged yet, so please use my copy for the moment.

This will get you a directory full of cropped numberplates that you can feed into the next step.







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
