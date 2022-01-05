# Recognizing ASL alphabet hand signs
 
#### Matthew Kwee, 3 Dec. 2021

## Abstract
In this project, I constructed and compared CNN and MLP networks in order to identify ASL hand signs.

I trained both models using a dataset from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet), and created a custom script for easy usage.


## Design
The goal of this project was to create a model that could identify ASL hand signs from images, as well as a live camera feed.


## Data
The Kaggle dataset contains 77,000 images, with around 3000 of each image class (A-Z, space, but minus J). The size, shape, format, and quality of the images vary greatly.

The 'J' sign is identical to the 'I' sign, but with the addition of a 'fishhook' movement. Because of the format of the dataset, identifying motion is impossible. The 'Z' sign also requires movement, but it is not otherwise identical to any other movement, so I left it in the set.

Many photos are of the same person and the same gesture, but at a slightly different angle, presumably to allow a network to generalize more effectively.

## Algorithms
Before using the images, I scaled each one to 64x64 in order to save storage space and allow a CNN to more easily create filters.

I divided the data into 80-20 train-test splits.

For the MLP model, I used MediaPipe to identify the locations of hand features, then trained the model with the (21,2) matrix that it yielded.


## Models
The first model I constructed was a CNN model.
After two hours of training, the CNN model's accuracy score was about 0.99 on test data, which made me suspect that it was overfitting and using the background to deduce the ASL hand sign. Testing the model with my own live camera data confirmed this - the CNN model was unable to correctly identify my hand signs, even with a blank background. As a result, I was forced to scrap the model and use a different approach.

The second model I used was an MLP classifier.

After feeding image data into Google's MediaPipe module to isolate the "skeleton" of the hand, I entered the resulting matrix into the MLP network. Training this model took only two minutes, and its accuracy score on test data was 0.942. When I tested it with a live camera feed, it was able to correctly identify gestures most of the time.


## Tools
OpenCV for converting image data into usable vectors
Google's MediaPipe for calculating hand "landmarks"
Pandas and NumPy for data storage and formatting
TensorFlow's Keras module for constructing neural networks
Matplotlib for data visualization


## Communication
In addition to my [slide presentation](https://docs.google.com/presentation/d/15aoP4n8LFIX6UCbp9nAwwszpL5KgcS2fGVurspfklR4/edit?usp=sharing), the app folder I created for demonstration can be found [here](https://github.com/MK38993/Metis-6---Deep-Learning/tree/main/python-demo).





