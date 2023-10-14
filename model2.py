import os
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt

# Define the source directory where the digit folders are located
source_dir = './numerals'  # Replace with the path to your images folder

# Define the destination directories for train, validation, and test splits
train_dir = './numerals_train'
validation_dir = './numerals_validation'
test_dir = './numerals_test'

# Create destination directories if they don't exist
for directory in [train_dir, validation_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize counters for each class
class_counts = {str(i): 0 for i in range(10)}

# Iterate through each digit folder (0 to 9)
for digit in range(10):
    digit_folder = os.path.join(source_dir, str(digit))
    
    # Get a list of image files in the folder
    image_files = [filename for filename in os.listdir(digit_folder) if filename.endswith('.jpg')]
    
    # Split the image files into train, validation, and test sets
    train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    # Copy images to the respective directories with modified names
    for filename in train_files:
        src_path = os.path.join(digit_folder, filename)
        dst_filename = f"{digit}_{filename}"  # Modified filename
        dst_path = os.path.join(train_dir, dst_filename)
        shutil.copy(src_path, dst_path)
        class_counts[str(digit)] += 1
    
    for filename in val_files:
        src_path = os.path.join(digit_folder, filename)
        dst_filename = f"{digit}_{filename}"  # Modified filename
        dst_path = os.path.join(validation_dir, dst_filename)
        shutil.copy(src_path, dst_path)
        class_counts[str(digit)] += 1
    
    for filename in test_files:
        src_path = os.path.join(digit_folder, filename)
        dst_filename = f"{digit}_{filename}"  # Modified filename
        dst_path = os.path.join(test_dir, dst_filename)
        shutil.copy(src_path, dst_path)
        class_counts[str(digit)] += 1

# Print the number of images in each class in the train, validation, and test splits
for digit in range(10):
    print(f"Class {digit}: {class_counts[str(digit)]} images")


def load_and_preprocess_images_from_directory(directory):
    print(f"Loading and Preprocessing Data from {directory}")
    images = []
    labels = []
    problematic_files = []  # List to store problematic file names
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            parts = filename.split('_')  # Split the filename into label and original name parts
            # print(parts)
            # print(len(parts))
            if len(parts) == 3:
                # print("inside len(parts)")
                label = int(parts[0])  # Extract the label from the filename
                try:
                    # Load the image and preprocess it (resize and normalize)
                    image = Image.open(os.path.join(directory, filename)).convert('L')  # Convert to grayscale
                    image = image.resize((32, 32))  # Resize to a consistent size
                    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

                    # Append the preprocessed image and label
                    images.append(image)
                    labels.append(label)
                    # print(f"image {image} and label {label}")
                except Exception as e:
                    # If there's an error while processing the image, log the filename
                    problematic_files.append(filename)
                    print(f"Error processing {filename}: {str(e)}")
    
    if problematic_files:
        print("Problematic files:", problematic_files)
    
    sys.stdout.flush()  # Force the output to be displayed immediately
    
    return np.array(images), np.array(labels)


def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# def softmax(self, x):
#     exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtracting max for numerical stability
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtracting max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)



# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights and biases with random values
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        self.train_losses = []  # List to store training loss values
        self.val_losses = []    # List to store validation loss values
        self.train_accuracies = []  # List to store training accuracy values
        self.val_accuracies = []    # List to store validation accuracy values
    
    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    # def sigmoid_derivative(self, x):
    #     return x * (1 - x)

    # Define the sigmoid activation function and its derivative
    
    def forward(self, x):
        # Flatten the input data to match the input layer's weights
        x = x.reshape(1, -1)

        # Calculate the output of the hidden layer with sigmoid activation
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Calculate the final output with softmax activation
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = softmax(self.final_input)  # Define the softmax function
        # self.final_output = softmax(self.final_input)


        return self.final_output


    # Modify the backward method in the NeuralNetwork class
    def backward(self, x, y, learning_rate):
        # Calculate the loss
        loss = y - self.final_output

        # Calculate the gradients and update the weights and biases for the output layer
        output_delta = loss
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        # Calculate the gradients and update the weights and biases for the hidden layer
        hidden_loss = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_loss * sigmoid_derivative(self.hidden_output)

        # Reshape x to match the shape of hidden_delta
        x_reshaped = x.reshape(1, -1)

        # Update the weights and biases for the input-hidden layer
        self.weights_input_hidden += x_reshaped.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate



    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            correct_train_predictions = 0

            for i in range(len(X_train)):
                x = X_train[i]
                target = y_train[i]

                # Forward pass
                output = self.forward(x)

                # Compute loss and update weights
                loss = np.mean(0.5 * (target - output) ** 2)  # Use mean squared error
                total_loss += loss
                self.backward(x, target, learning_rate)

                # Count correct predictions for training accuracy
                if np.argmax(output) == np.argmax(target):
                    correct_train_predictions += 1

            # Calculate average training loss and accuracy for the epoch
            avg_train_loss = total_loss / len(X_train)
            train_accuracy = correct_train_predictions / len(X_train) * 100
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)

            # Validation phase
            total_val_loss = 0
            correct_val_predictions = 0

            for i in range(len(X_val)):
                x_val = X_val[i]
                target_val = y_val[i]

                # Forward pass
                val_output = self.forward(x_val)

                # Compute validation loss
                val_loss = np.mean(0.5 * (target_val - val_output) ** 2)
                total_val_loss += val_loss

                # Count correct predictions for validation accuracy
                if np.argmax(val_output) == np.argmax(target_val):
                    correct_val_predictions += 1

            # Calculate average validation loss and accuracy for the epoch
            avg_val_loss = total_val_loss / len(X_val)
            val_accuracy = correct_val_predictions / len(X_val) * 100
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)

            # Print progress for each epoch
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        print("Training complete.")

    def predict(self, x):
        # Make predictions for a single input
        return np.argmax(self.forward(x))

# Load and preprocess the images
X_train, y_train = load_and_preprocess_images_from_directory(train_dir)
X_val, y_val = load_and_preprocess_images_from_directory(validation_dir)
X_test, y_test = load_and_preprocess_images_from_directory(test_dir)

# i = 0
# for filename in os.listdir(train_dir):
#     label = os.path.basename(filename)[0]
#     print(f"TRAIN: Filename: {filename}, First Character: {label}")
#     print(y_train[i])
#     i = i + 1
#     # y_train.append(label)
# i = 0
# # Assuming X_val is a list of file paths
# for filename in os.listdir(validation_dir):
#     label = os.path.basename(filename)[0]
#     print(f"VAL: Filename: {filename}, First Character: {label}")
#     print(y_val[i])
#     i = i + 1
#     # y_val.append(label)
# i = 0
# # Assuming X_test is a list of file paths
# for filename in os.listdir(test_dir):
#     label = os.path.basename(filename)[0]
#     print(f"TEST: Filename: {filename}, First Character: {label}")
#     print(y_test[i])
#     i = i + 1
#     # y_test.append(label)

# Define the neural network architecture
input_size = 1024
hidden_size = 64  # Adjust as needed
output_size = 10  # 10 classes for digits 0-9

# Initialize and train the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)
epochs = 20  # Adjust as needed
learning_rate = 0.001  # Adjust as needed
nn.train(X_train, y_train, X_val, y_val, epochs, learning_rate)

# Evaluate the model on the test set
correct_test_predictions = 0
total_test_samples = len(X_test)

for i in range(total_test_samples):
    x_test = X_test[i]
    target_test = y_test[i]
    prediction = nn.predict(x_test)

    if prediction == np.argmax(target_test):
        correct_test_predictions += 1

test_accuracy = correct_test_predictions / total_test_samples * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(nn.train_losses, label='Training Loss')
plt.plot(nn.val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(nn.train_accuracies, label='Training Accuracy')
plt.plot(nn.val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()


# Prediction phase
print("Making predictions on the test set...")
predictions = model.predict(X_test)
correct_predictions = np.sum(np.array(predictions) == np.array(y_test))
print(predictions)
print(correct_predictions)
print("Predictions complete.")

# import random

# # Function to display images and predictions
# def display_images_with_predictions(images, true_labels, predictions, class_names):
#     num_samples = len(images)
#     sample_indices = random.sample(range(num_samples), 3)  # Select 3 random samples
    
#     plt.figure(figsize=(12, 4))
    
#     for i, index in enumerate(sample_indices):
#         plt.subplot(1, 3, i + 1)
#         plt.imshow(images[index], cmap='gray')
#         plt.title(f"True: {class_names[true_labels[index]]}\nPredicted: {class_names[predictions[index]]}")
#         plt.axis('off')

# # List of class names (replace with your class names if needed)
# class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"]

# # Display images with predictions
# display_images_with_predictions(X_test, y_test, predictions, class_names)
# plt.show()


# # Plot the learning curves (loss and accuracy)
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(model.train_losses, label='Training Loss')
# plt.plot(model.val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training and Validation Loss')

# plt.subplot(1, 2, 2)
# plt.plot(model.train_accuracies, label='Training Accuracy')
# plt.plot(model.val_accuracies, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend()
# plt.title('Training and Validation Accuracy')

# plt.tight_layout()
# plt.show()

# # You can now use the 'predictions' variable for further analysis or evaluation

