# Homeassignment3
#Autoencoder for Image Reconstruction
#Introduction
This project demonstrates how to implement a fully connected autoencoder using TensorFlow and Keras. The autoencoder learns to compress and then reconstruct images from the MNIST dataset. It is a classic application of unsupervised learning for dimensionality reduction and feature learning.
#Requirements
To run this notebook, install the following packages:
pip install numpy matplotlib tensorflow
#Code Overview
The notebook includes the following sections:
Importing Libraries: Load essential Python libraries such as TensorFlow, NumPy, and Matplotlib.
Data Preparation: Load the MNIST dataset and normalize/flatten it.
Model Architecture:
Encoder: Input layer (784 nodes) → Dense(32).
Decoder: Dense(784) with sigmoid activation.
Training: The autoencoder is trained using binary cross-entropy loss and the Adam optimizer.
Visualization: After training, original and reconstructed images are plotted side by side.
Usage
Clone the repository or download the notebook.
Install the required libraries.
Run the Jupyter notebook:
jupyter notebook "Autoencoder_Image_Reconstruction.ipynb"
ey Functions
Model() – Keras functional API to create the autoencoder.
Dense() – Used for creating encoder and decoder layers.
fit() – Trains the model on input data.
matplotlib.pyplot – Visualizes original vs reconstructed images.
