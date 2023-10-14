import numpy as np
import os
import sys
from PIL import Image
from tensorflow.keras.utils import to_categorical

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

def forward_propagate(self, X):
    # Feed forward input X through a network
    H = self.sigmoid(np.dot(X, self.W1))
    Y = self.sigmoid(np.dot(H, self.W2))
    return H, Y

def back_propagate(self, X, H, Y, y_true):
    # Back propagate errors and update weights
    m = X.shape[0]
    error = Y - y_true
    dW2 = (1/m) * np.dot(H.T, error)
    dH = np.dot(error, self.W2.T) * self.sigmoid_derivative(H)
    dW1 = (1/m) * np.dot(X.T, dH)
    return dW1, dW2

def initialize_weights(self, input_size, hidden_size, output_size):
    # Initialize weights randomly with a mean of 0
    W1 = np.random.randn(input_size, hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    return W1, W2

def update_weights(self, dW1, dW2, learning_rate):
    # Update weights using gradient descent
    self.W1 -= learning_rate * dW1
    self.W2 -= learning_rate * dW2

# Separate the Neuron and Model classes

class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1, self.W2 = self.initialize_weights(input_size, hidden_size, output_size)

    def sigmoid(self, x):
        # Activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        return x * (1 - x)

    def initialize_weights(self, input_size, hidden_size, output_size):
        # Initialize weights randomly with a mean of 0
        W1 = np.random.randn(input_size, hidden_size)
        W2 = np.random.randn(hidden_size, output_size)
        return W1, W2

    def forward_propagate(self, X):
        H = self.sigmoid(np.dot(X, self.W1))
        Y = self.sigmoid(np.dot(H, self.W2))
        return H, Y

    def back_propagate(self, X, Y, y_true):
        m = X.shape[0]
        error = Y - y_true
        dW2 = (1 / m) * np.dot(X.T, error)  # Update dW2 accordingly
        dH = np.dot(error, self.W2.T) * self.sigmoid_derivative(Y)
        dW1 = (1 / m) * np.dot(X.T, dH)  # Update dW1 accordingly
        return dW1, dW2

    # def back_propagate(self, X, H, Y, y_true):
    #   m = X.shape[0]
    # #   print(f"X_batch size: {m}")
    # #   y_true_rezied = y_true
    # #   print(f"y_true_rezied: {y_true_rezied.shape}")
    # #   y_true_rezied = y_true
    # #   print(f"y_true_rezied [0]: {y_true_rezied.shape[0]}")
    # #   y_true_rezied = y_true[0]
    # #   print(f"y_true_rezied0: {y_true_rezied.shape}")
    # #   y_true_rezied = y_true[1]
    # #   print(f"y_true_rezied: {y_true_rezied.shape}")
    # #   y_true_rezied = y_true[2]
    # #   print(f"y_true_rezied2: {y_true_rezied.shape}")
    # #   print(f"self.W2.shape[1]: {self.W2.shape[1]}")
    # #   y_true_rezied= y_true_rezied.reshape(m, self.W2.shape[1])

    #   y_true_rezied = y_true
    #   print(f"y_true_rezied.shape: {y_true_rezied.shape}")
    #   # Ensure that y_true has the correct shape (batch_size, num_classes)
    # #   if y_true_rezied.shape!= (m, self.W2.shape[1]):
    # #     #   print(f"y_true_rezied should have shape ({m}, {self.W2.shape[1]}), but got shape {y_true_rezied.shape}")
    # #     #   y_true_rezied.reshape(m)
    # #       print(f"y_true_rezied should have shape ({m}, {self.W2.shape[1]}), but got shape {y_true_rezied.shape}")
    # #       raise ValueError(f"y_true_rezied should have shape ({m}, {self.W2.shape[1]}), but got shape {y_true_rezied.shape}")
    #   print(f"y_true_rezied should have shape ({m}, {self.W2.shape[1]}), but got shape {y_true_rezied.shape}")
    #   error = Y - y_true_rezied
    #   dW2 = (1 / m) * np.dot(H.T, error)
    #   dH = np.dot(error, self.W2.T) * self.sigmoid_derivative(H)
    #   dW1 = (1 / m) * np.dot(X.T, dH)
    #   return dW1, dW2


    def update_weights(self, dW1, dW2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2

    def optimize(self, X, y_true, num_epochs, learning_rate, batch_size):

        for epoch in range(num_epochs):
            print(f"epoch {epoch}")
            shuffled_indices = np.random.permutation(X.shape[0])
            X = X[shuffled_indices]
            y_true = y_true[shuffled_indices]

            num_batches = X.shape[0] // batch_size
            # print(f"num_batches for X.shape[0]: {num_batches}")
            # num_batches = batch_size/X.shape[0]
            # print(f"num_batches for batch_size/X.shape[0]: {num_batches}")
            # for batch_index in range(0, X.shape[0], batch_size):
            #     start_index = batch_index
            #     end_index = start_index + batch_size
            #     X_batch = X[start_index:end_index]
            #     y_batch = y_true[start_index:end_index]
            for batch_index in range(int(num_batches)):
                start_index = batch_index * batch_size
                end_index = start_index + batch_size
                X_batch = X[start_index:end_index]
                y_batch = y_true[start_index:end_index]

                H, Y = self.forward_propagate(X_batch)
                dW1, dW2 = self.back_propagate(X_batch, Y, y_batch)
                self.update_weights(dW1, dW2, learning_rate)


                Y = self.forward_propagate(X_batch)
                print(f"X_batch size: {X_batch.shape}, Y: {Y.shape}, y_batch: {y_batch.shape}")
                # print(f"X_batch size: {X_batch.shape}, H: {H.shape}, Y: {Y.shape}, y_batch: {y_batch.shape}")
                # dW1, dW2 = self.back_propagate(X_batch, H, Y, y_batch)
                # self.update_weights(dW1, dW2, learning_rate)
                dW1, dW2 = self.back_propagate(X_batch, Y, y_batch)
                self.update_weights(dW1, dW2, learning_rate)

# Define the source directory where the digit folders are located
source_dir = './numerals'  # Replace with the path to your images folder

# Define the destination directories for train, validation, and test splits
train_dir = './numerals_train'
validation_dir = './numerals_validation'
test_dir = './numerals_test'
def load_and_preprocess_images_from_directory(directory, class_labels, output_size):
    images = []
    labels = []
    problematic_files = []  # List to store problematic file names
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            parts = filename.split('_')  # Split the filename into label and original name parts
            if len(parts) == 3:
                label = int(parts[0])  # Extract the label from the filename
                try:
                    # Load the image
                    image = Image.open(os.path.join(directory, filename))
                    image_array = np.array(image)

                    # Check if the image is RGB or grayscale
                    if len(image_array.shape) == 2:
                        print(f"Image '{filename}' is grayscale.")
                    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                        print(f"Image '{filename}' is RGB.")
                    else:
                        print(f"Image '{filename}' has an unexpected number of channels.")

                    # Get the original size of the image
                    original_size = image_array.shape[:2]
                    print(f"Original size of '{filename}': {original_size}")

                    # Normalize pixel values to [0, 1]
                    image_array = image_array / 255.0

                    # Append the preprocessed image and label
                    images.append(image_array)

                    # Ensure that y_batch is one-hot encoded with the correct shape
                    y_batch_one_hot = np.zeros(output_size)
                    y_batch_one_hot[label] = 1  # Set the appropriate index to 1
                    labels.append(y_batch_one_hot)
                except Exception as e:
                    # If there's an error while processing the image, log the filename
                    problematic_files.append(filename)
                    print(f"Error processing {filename}: {str(e)}")

    if problematic_files:
        print("Problematic files:", problematic_files)

    sys.stdout.flush()  # Force the output to be displayed immediately
    print(f"Number of images: {len(images)}\nNumber of labels: {len(labels)}")
    return np.array(images), np.array(labels)




# # Load and preprocess the images from train, validation, and test directories
# X_train, y_train = load_and_preprocess_images_from_directory(train_dir)
# X_test, y_test = load_and_preprocess_images_from_directory(test_dir)

# Define the class labels and output size
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Adjust this based on your dataset
output_size = len(class_labels)

# Load and preprocess the images from train, validation, and test directories
X_train, y_train = load_and_preprocess_images_from_directory(train_dir, class_labels, output_size)
X_test, y_test = load_and_preprocess_images_from_directory(test_dir, class_labels, output_size)

# # One-hot encode labels
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)
# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# Create model
model = Model(input_size=X_train.shape[1], hidden_size=1024, output_size=10)

# Train model
model.optimize(X_train, y_train, num_epochs=10, learning_rate=0.01, batch_size=64)

# Evaluate model
H, Y = model.forward_propagate(X_test)
predictions = np.argmax(Y, axis=1)
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print
