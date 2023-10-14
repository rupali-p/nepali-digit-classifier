import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score
import seaborn as sns


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

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

        self.train_losses = []  # Store training loss for each epoch
        self.val_losses = []  # Store validation loss for each epoch
        self.train_accuracies = []  # Store training accuracy for each epoch
        self.val_accuracies = []  # Store validation accuracy for each epoch

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_probabilities = softmax(self.output)
        return self.output_probabilities

    def backward(self, X, y):
        batch_size = X.shape[0]
        d_output = self.output_probabilities - y
        d_hidden = np.dot(d_output, self.weights_hidden_output.T) * relu_derivative(self.hidden_output)

        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, d_output) / batch_size
        self.bias_output -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True) / batch_size
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, d_hidden) / batch_size
        self.bias_hidden -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True) / batch_size

    def train(self, X_train, y_train, X_val, y_val, epochs):
        for epoch in range(epochs):
            # Training
            output = self.forward(X_train)
            self.backward(X_train, y_train)
            train_loss = -np.sum(y_train * np.log(output)) / len(X_train)
            self.train_losses.append(train_loss)

            # Validation
            val_output = self.forward(X_val)
            val_loss = -np.sum(y_val * np.log(val_output)) / len(X_val)
            self.val_losses.append(val_loss)

            # Calculate and store training accuracy
            train_predictions = self.predict(X_train)
            train_accuracy = np.mean(train_predictions == np.argmax(y_train, axis=1))
            self.train_accuracies.append(train_accuracy)

            # Calculate and store validation accuracy
            val_predictions = self.predict(X_val)
            val_accuracy = np.mean(val_predictions == np.argmax(y_val, axis=1))
            self.val_accuracies.append(val_accuracy)

            # Print training and validation metrics
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Function to load and preprocess images from a directory
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

# Load and preprocess the images from train, validation, and test directories
source_dir = './numerals'
train_dir = './numerals_train'
validation_dir = './numerals_validation'
test_dir = './numerals_test'
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
output_size = len(class_labels)

X_train, y_train = load_and_preprocess_images_from_directory(train_dir, output_size)
X_val, y_val = load_and_preprocess_images_from_directory(validation_dir, output_size)
X_test, y_test = load_and_preprocess_images_from_directory(test_dir, output_size)

# Assuming your images are grayscale and stored as 28x28 images
# Flatten and preprocess the input data
X_train = X_train.reshape(-1, 784)
X_train = X_train / 255.0

X_val = X_val.reshape(-1, 784)
X_val = X_val / 255.0

X_test = X_test.reshape(-1, 784)
X_test = X_test / 255.0

# Create and train the neural network
input_size = 784
hidden_size = 2048
output_size = 10
learning_rate = 0.01
epochs = 10

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
nn.train(X_train, y_train, X_val, y_val, epochs)

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(nn.train_losses, label="Training Loss")
plt.plot(nn.val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot training and validation accuracies
plt.figure(figsize=(10, 5))
plt.plot(nn.train_accuracies, label="Training Accuracy")
plt.plot(nn.val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Make predictions on test data
predictions = nn.predict(X_test)
print("Predictions:", predictions)

# Calculate and display accuracy for the test set
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Calculate confusion matrix and metrics for the test set
test_true_labels = np.argmax(y_test, axis=1)
test_confusion_matrix = confusion_matrix(test_true_labels, predictions)
test_classification_report = classification_report(test_true_labels, predictions, target_names=[str(label) for label in class_labels])

# Print and display the confusion matrix and metrics for the test set
print("Confusion Matrix (Test Set):")
print(test_confusion_matrix)

print("\nClassification Report (Test Set):")
print(test_classification_report)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Set)')
plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 5))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(output_size):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], nn.forward(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curves
plt.figure(figsize=(10, 5))
precision = dict()
recall = dict()
average_precision = dict()
for i in range(output_size):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], nn.forward(X_test)[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], nn.forward(X_test)[:, i])
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AP = {average_precision[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="lower left")
plt.show()
