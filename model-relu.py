import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the softmax function for the output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Experiment with a different activation function (e.g., ReLU)
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


# Define a neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

   # Modify the forward and backward functions to use ReLU
    def forward(self, X):
        # Feedforward operation with ReLU activation for the hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_probabilities = softmax(self.output)
        return self.output_probabilities

    def backward(self, X, y):
        # Backpropagation with ReLU derivative
        batch_size = X.shape[0]
        d_output = self.output_probabilities - y
        d_hidden = np.dot(d_output, self.weights_hidden_output.T) * relu_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, d_output) / batch_size
        self.bias_output -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True) / batch_size
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, d_hidden) / batch_size
        self.bias_hidden -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True) / batch_size

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            loss = -np.sum(y * np.log(output)) / len(X)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Generate some synthetic data (you should load your dataset here)
np.random.seed(0)

# Define the class labels and output size
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Adjust this based on your dataset
output_size = len(class_labels)
# Define the source directory where the digit folders are located
source_dir = './numerals'  # Replace with the path to your images folder

# Define the destination directories for train, validation, and test splits
train_dir = './numerals_train'
validation_dir = './numerals_validation'
test_dir = './numerals_test'
def load_and_preprocess_images_from_directory(directory, output_size):
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

                    # # Check if the image is RGB or grayscale
                    # if len(image_array.shape) == 2:
                    #     print(f"Image '{filename}' is grayscale.")
                    # elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    #     print(f"Image '{filename}' is RGB.")
                    # else:
                    #     print(f"Image '{filename}' has an unexpected number of channels.")

                    # Get the original size of the image
                    original_size = image_array.shape[:2]
                    # print(f"Original size of '{filename}': {original_size}")

                    # Normalize pixel values to [0, 1]
                    image_array = image_array / 255.0

                    # Append the preprocessed image and label
                    images.append(image_array)

                    # Ensure that y_batch is one-hot encoded with the correct shape
                    y_batch_one_hot = np.zeros(output_size)
                    y_batch_one_hot[label] = 1  # Set the appropriate index to 1
                    # For example, if label is 3 (indicating the image belongs to class 3), the one-hot encoding vector would look like this:
                    # y_batch_one_hot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
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
# Load and preprocess the images from train, validation, and test directories
# X_train, y_train = load_and_preprocess_images_from_directory(train_dir, class_labels, output_size)
# test_data, y_test = load_and_preprocess_images_from_directory(test_dir, class_labels, output_size)

X_train, y_train = load_and_preprocess_images_from_directory(train_dir, output_size)
# X_val, y_val = load_and_preprocess_images_from_directory(validation_dir, class_labels, output_size)
X_test, y_test = load_and_preprocess_images_from_directory(test_dir, output_size)

# Assuming your images are grayscale and stored as 28x28 images
# Flatten and preprocess the input data
X_train = X_train.reshape(-1, 784)  # Reshape to (number_of_samples, 784)
X_train = X_train / 255.0  # Normalize pixel values to [0, 1]

# Similarly, preprocess the test data
X_test = X_test.reshape(-1, 784)
X_test = X_test / 255.0



# Create and train the neural network
input_size = 784  # 28x28 = 784 pixels
hidden_size = 2048 # 512 # 1024
output_size = 10  # 10 classes
learning_rate = 0.01 # 0.01 seems good, 0.1 is not the best
epochs = 1

# # Create and train the neural network
# input_size = 784  # 28x28 = 784 pixels
# hidden_size = 2048 # 512 # 1024
# output_size = 10  # 10 classes
# learning_rate = 0.01 # 0.01 seems good, 0.1 is not the best
# epochs = 1

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
nn.train(X_train, y_train, epochs)

# Make predictions on new data
predictions = nn.predict(X_test)
print("Predictions:", predictions)

# Calculate and display accuracy
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"Accuracy: {accuracy * 100:.2f}%")

# Get the list of image names (assuming your filenames are available)
image_names = [filename for filename in os.listdir(test_dir) if filename.endswith('.jpg')]

# Visualize some results along with image names, real labels, and predictions
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    
    # Get the image name, real label, and prediction
    image_name = image_names[i]
    real_label = np.argmax(y_test[i])  # Convert one-hot encoding to label
    prediction = predictions[i]
    
    # Display image name, real label, and prediction as the title with padding
    padded_name = f"{image_name}"  # Adjust the padding width as needed
    padded_real_label = f"Real: {real_label}"
    padded_prediction = f"Pred: {prediction}"
    title_text = f"{padded_name}\n{padded_real_label} {padded_prediction}"
    
    plt.title(title_text, fontsize=8)  # Adjust fontsize as needed
    plt.axis('off')

# Add an overall title with accuracy
accuracy_text = f"Accuracy: {accuracy * 100:.2f}%\n\nHidden Size: {hidden_size}, Learning Rate: {learning_rate}, Epochs: {epochs}"
plt.suptitle(accuracy_text, fontsize=10, y=0.995)  # Adjust fontsize and position as needed

plt.tight_layout()
plt.show()