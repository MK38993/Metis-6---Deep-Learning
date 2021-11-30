# Metis Project 6: ASL Alphabet - MVP
 
The goal of this project is to recognize hand signs from the ASL alphabet.
In order to begin analysis, I divided my data into 80-20 train/test groups. I then created a convolutional neural network, and trained it based on a small fraction of the training data. I then scored the model using 2048 randomly selected test images.

The model's accuracy score was 0.8941 on validation data, and 0.8535 on test data.

I plan to construct another CNN model using all training data, then a third model utilizing the MediaPipe library's "hand landmark" recognition software, which creates an outline of the hand's key points. 


EDIT: Tuesday 10:30 AM - I trained a more complex model overnight, which uses my entire training dataset. Its accuracy score on validation data was 0.9887, and 0.9912 on test data.