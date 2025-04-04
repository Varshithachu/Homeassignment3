# Homeassignment3
# Autoencoder for Image Reconstruction
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
#Usage
Clone the repository or download the notebook.
Install the required libraries.
Run the Jupyter notebook:
jupyter notebook "Autoencoder_Image_Reconstruction.ipynb"
#key Functions
Model() – Keras functional API to create the autoencoder.
Dense() – Used for creating encoder and decoder layers.
fit() – Trains the model on input data.
matplotlib.pyplot – Visualizes original vs reconstructed images.


# Denoising Autoencoder for Image Reconstruction
#Introduction
This project demonstrates a denoising autoencoder that learns to reconstruct clean images from noisy inputs. This technique is useful for learning robust data representations and removing noise from corrupted inputs.
#Requirements
Install the required packages using:
pip install numpy matplotlib tensorflow
#Code Overview
The notebook contains the following sections:
Data Loading: Load MNIST and normalize pixel values.
Adding Noise: Gaussian noise (mean=0, std=0.5) is added to the input images.
Model Architecture: The same encoder-decoder architecture as a basic autoencoder, trained with noisy inputs and clean targets.
Training: The model is trained to reconstruct clean images from noisy versions.
Visualization: Compares noisy, reconstructed, and original images.
Comparison: Demonstrates how denoising autoencoders outperform basic ones in noisy environments.
#Key Functions
np.random.normal() – Adds Gaussian noise to images.
Model() – Keras functional API to define the network.
fit() – Trains the autoencoder on noisy data.
matplotlib.pyplot – Used to visualize results.
#Usage
jupyter notebook "Denoising_Autoencoder.ipynb"

# RNN for Character-Level Text Generation
#Introduction
This notebook builds a Recurrent Neural Network (RNN) using LSTM layers to generate text. The model learns to predict the next character in a sequence using Shakespeare's text and is capable of generating new, similar-sounding text after training.
#Requirements
Install necessary dependencies:
pip install numpy tensorflow
#Code Overview
Dataset Loading: Downloads Shakespeare’s complete works from TensorFlow Datasets.
Preprocessing: Converts text into one-hot encoded character sequences.
Model Architecture:
LSTM with 128 units.
Dense layer with softmax activation for next-character prediction.
Training: Uses categorical cross-entropy loss and the Adam optimizer.
Generation: Predicts the next character iteratively using a helper function and adjustable temperature parameter.
#Temperature Scaling
Temperature controls the randomness of predictions during sampling:
Low (e.g., 0.2): More predictable and repetitive output.
High (e.g., 1.0): More creative and diverse, but less coherent output.
Temperature helps balance between creativity and accuracy in generated text.
#Usage
Run the notebook with:
jupyter notebook "Text_Generation_RNN.ipynb"

# Sentiment Classification Using LSTM (IMDB Reviews)
#Introduction
This notebook demonstrates how to use LSTM-based RNNs for binary sentiment classification using the IMDB movie review dataset. The model is trained to determine whether a review is positive or negative based on its text content.
#Requirements
pip install tensorflow scikit-learn matplotlib seaborn
#Code Overview
Dataset Loading: The IMDB dataset is pre-tokenized and limited to the top 10,000 most frequent words.
Preprocessing: Reviews are padded to a fixed length (200 words) for uniform input size.
Model Architecture:
Embedding layer to convert integers to word vectors.
LSTM layer with 64 units.
Output layer with sigmoid activation for binary classification.
Training: Binary cross-entropy loss with Adam optimizer is used.
Evaluation:
A confusion matrix shows the count of true vs predicted classes.
A classification report includes precision, recall, F1-score, and accuracy.
#Precision vs. Recall Tradeoff
In sentiment classification:
Precision ensures that when the model predicts "positive", it's usually correct. This is crucial when false positives are costly (e.g., auto-approving user reviews).
Recall ensures that most actual positives are identified, which is important when missing true positives is more critical (e.g., detecting negative feedback).
The F1-score balances both, offering a better metric when there’s a class imbalance.
#Usage
jupyter notebook "Sentiment_Classification_LSTM.ipynb"



