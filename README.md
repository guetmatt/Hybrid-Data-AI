This is an implementation for an image classification and synthetic image generation task for a university course. The tasks include classifying images into images of cars, bikes, and busses. Additionally, synthetic images have been created (see directory ```synthetic_images```), trying to enhance the training process of a classifier model. Several experiments have been conducted. Further details can be found in the accompanying paper. 

# Python Dependencies
- tensorflow version >= 2.17.0
- NumPy
- PIL
- pathlib
- joblib
- glob
- imageio
- matplotlib
- time


# Python Files

## image_classification_cnn.py
Create, train and test a cnn model (Convolutional Neural Network) on an image classification task using images from the 'data' directory. Can also be used to load trained models from the 'models' directory. Different experiment setups are further detailed in the accompanying paper.

## image_generation_dcgan.py
Create and train a dcgan model (Deep Convolutional Generative Adversarial Network) using images from the 'dataset_cars_bikes' directory. Can also be used to generate synthetic images. A model trained car images can be loaded from the 'training_checkpoints' directory.
