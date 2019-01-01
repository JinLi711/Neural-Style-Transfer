# Neural Style Transfer

This repository contains a script that allows one to combine the style of one image to a base image.

# Model

Check out Context.txt

## Example

Base Image:

<img src="https://github.com/JinLi711/Neural-Style-Transfer/blob/master/images/target/cat.jpg" alt="Cat" width="400" height="400">

Style: 

<img src="https://github.com/JinLi711/Neural-Style-Transfer/blob/master/images/style/trippy.jpg" alt="Style" width="400" height="400">

Generated Image:

<img src="https://github.com/JinLi711/Neural-Style-Transfer/blob/master/images/generated/cat_trippy_at_iteration_4.png" alt="Generated Image" width="400" height="400">

## Things To Work On

I could train the VGG model to incorporate a style, so every time I want to incorporate a certain style, I would only have to run the image through once and the results would be instant.

## Acknowledgements

The code in the repository is based off of the neural style transfer from Keras.
https://github.com/keras-team/keras
