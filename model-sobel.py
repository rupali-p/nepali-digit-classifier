# import os
# import sys
# import numpy as np
# from PIL import Image
# from sklearn.model_selection import train_test_split
# import shutil
# import matplotlib.pyplot as plt

# # Define the source directory where the digit folders are located
# source_dir = './numerals'  # Replace with the path to your images folder

# # Define the destination directories for train, validation, and test splits
# train_dir = './numerals_train'
# validation_dir = './numerals_validation'
# test_dir = './numerals_test'

# # Create destination directories if they don't exist
# for directory in [train_dir, validation_dir, test_dir]:
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# # Initialize counters for each class
# class_counts = {str(i): 0 for i in range(10)}

# # Iterate through each digit folder (0 to 9)
# for digit in range(10):
#     digit_folder = os.path.join(source_dir, str(digit))
    
#     # Get a list of image files in the folder
#     image_files = [filename for filename in os.listdir(digit_folder) if filename.endswith('.jpg')]
    
#     # Split the image files into train, validation, and test sets
#     train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
#     val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
#     # Copy images to the respective directories with modified names
#     for filename in train_files:
#         src_path = os.path.join(digit_folder, filename)
#         dst_filename = f"{digit}_{filename}"  # Modified filename
#         dst_path = os.path.join(train_dir, dst_filename)
#         shutil.copy(src_path, dst_path)
#         class_counts[str(digit)] += 1
    
#     for filename in val_files:
#         src_path = os.path.join(digit_folder, filename)
#         dst_filename = f"{digit}_{filename}"  # Modified filename
#         dst_path = os.path.join(validation_dir, dst_filename)
#         shutil.copy(src_path, dst_path)
#         class_counts[str(digit)] += 1
    
#     for filename in test_files:
#         src_path = os.path.join(digit_folder, filename)
#         dst_filename = f"{digit}_{filename}"  # Modified filename
#         dst_path = os.path.join(test_dir, dst_filename)
#         shutil.copy(src_path, dst_path)
#         class_counts[str(digit)] += 1

# # Print the number of images in each class in the train, validation, and test splits
# for digit in range(10):
#     print(f"Class {digit}: {class_counts[str(digit)]} images")

# # Define the function to load and preprocess images from a directory
# # def load_and_preprocess_images_from_directory(directory):
# #     print(f"Loading and Preprocessing Data from {directory}")
# #     images = []
# #     labels = []
# #     for filename in os.listdir(directory):
# #         if filename.endswith('.jpg'):
# #             parts = filename.split('_')  # Split the filename into label and original name parts
# #             if len(parts) == 2:
# #                 label = int(parts[0])  # Extract the label from the filename
# #                 # Load the image and preprocess it (resize and normalize)
# #                 image = Image.open(os.path.join(directory, filename)).convert('L')  # Convert to grayscale
# #                 image = image.resize((32, 32))  # Resize to a consistent size
# #                 image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

# #                 # Append the preprocessed image and label
# #                 images.append(image)
# #                 labels.append(label)
# #                 print(f"image {image} and label {label}")
# #                 sys.stdout.flush()  # Force the output to be displayed immediately
# #     print(f"image {images} and label {labels}")
# #     return np.array(images), np.array(labels)

# import sys

# def load_and_preprocess_images_from_directory(directory):
#     print(f"Loading and Preprocessing Data from {directory}")
#     images = []
#     labels = []
#     problematic_files = []  # List to store problematic file names
#     for filename in os.listdir(directory):
#         if filename.endswith('.jpg'):
#             parts = filename.split('_')  # Split the filename into label and original name parts
#             # print(parts)
#             # print(len(parts))
#             if len(parts) == 3:
#                 # print("inside len(parts)")
#                 label = int(parts[0])  # Extract the label from the filename
#                 try:
#                     # Load the image and preprocess it (resize and normalize)
#                     image = Image.open(os.path.join(directory, filename)).convert('L')  # Convert to grayscale
#                     image = image.resize((32, 32))  # Resize to a consistent size
#                     image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

#                     # Append the preprocessed image and label
#                     images.append(image)
#                     labels.append(label)
#                     # print(f"image {image} and label {label}")
#                 except Exception as e:
#                     # If there's an error while processing the image, log the filename
#                     problematic_files.append(filename)
#                     print(f"Error processing {filename}: {str(e)}")
    
#     if problematic_files:
#         print("Problematic files:", problematic_files)
    
#     sys.stdout.flush()  # Force the output to be displayed immediately
    
#     return np.array(images), np.array(labels)


# # Load and preprocess the images from train, validation, and test directories
# X_train, y_train = load_and_preprocess_images_from_directory(train_dir)
# X_val, y_val = load_and_preprocess_images_from_directory(validation_dir)
# X_test, y_test = load_and_preprocess_images_from_directory(test_dir)
# # y_train = []
# # y_val = []
# # y_test = []

# print("STOP")
# # Assuming X_train is a list of file paths
# i = 0
# for filename in os.listdir(train_dir):
#     label = os.path.basename(filename)[0]
#     # print(f"TRAIN: Filename: {filename}, First Character: {label}")
#     # print(y_train[i])
#     i = i + 1
#     # y_train.append(label)
# i = 0
# # Assuming X_val is a list of file paths
# for filename in os.listdir(validation_dir):
#     label = os.path.basename(filename)[0]
#     # print(f"VAL: Filename: {filename}, First Character: {label}")
#     # print(y_val[i])
#     i = i + 1
#     # y_val.append(label)
# i = 0
# # Assuming X_test is a list of file paths
# for filename in os.listdir(test_dir):
#     label = os.path.basename(filename)[0]
#     # print(f"TEST: Filename: {filename}, First Character: {label}")
#     # print(y_test[i])
#     i = i + 1
#     # y_test.append(label)


# # Print the number of images in each class in the train, validation, and test splits
# for digit in range(10):
#     print(f"Class {digit}: {class_counts[str(digit)]} images")

# # Define the sigmoid activation function and its derivative
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# # Define the Sobel operator kernels for gradient calculation
# sobel_x = np.array([[-1, 0, 1],
#                     [-2, 0, 2],
#                     [-1, 0, 1]])

# sobel_y = np.array([[-1, -2, -1],
#                     [0, 0, 0],
#                     [1, 2, 1]])

# # Define the neural network class
# class NeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size):
#         # Initialize the weights and biases with random values
#         print("Initializing NN")
#         self.weights_input_hidden = np.random.rand(input_size, hidden_size)
#         self.bias_hidden = np.zeros((1, hidden_size))
#         self.weights_hidden_output = np.random.rand(hidden_size, output_size)
#         self.bias_output = np.zeros((1, output_size))
#         self.train_losses = []  # List to store training loss values
#         self.val_losses = []    # List to store validation loss values
#         self.train_accuracies = []  # List to store training accuracy values
#         self.val_accuracies = []    # List to store validation accuracy values

#     def forward(self, x):
#         # Flatten the input data to match the input layer's weights
#         x = x.reshape(1, -1)

#         # Calculate the output of the hidden layer
#         self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
#         self.hidden_output = sigmoid(self.hidden_input)

#         # Calculate the final output
#         self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
#         self.final_output = sigmoid(self.final_input)

#         return self.final_output

#     def backward(self, x, y, learning_rate):
#         # Calculate the loss
#         loss = y - self.final_output

#         # Calculate the gradients and update the weights and biases
#         output_delta = loss * sigmoid_derivative(self.final_output)
#         hidden_loss = output_delta.dot(self.weights_hidden_output.T)
#         hidden_delta = hidden_loss * sigmoid_derivative(self.hidden_output)

#         self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
#         self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
#         self.weights_input_hidden += x.reshape(-1, 1).dot(hidden_delta.reshape(1, -1)) * learning_rate
#         self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

#     def train(self, X, y, epochs, learning_rate):
#         print("Training")
#         # Training phase
#         print("Training the neural network...")
#         for epoch in range(epochs):
#             total_loss = 0
#             correct_predictions = 0
#             print(f"len of train {len(X)}")
#             for i in range(len(X)):
#                 x = X[i]
#                 target = y[i]

#                 # Forward pass
#                 output = model.forward(x)

#                 # Compute loss
#                 loss = np.mean(0.5 * (target - output) ** 2)
#                 total_loss += loss
#                 print(f"loss: {loss} and total loss: {total_loss}")

#                 # Backward pass
#                 model.backward(x, target, learning_rate)

#                 # Count correct predictions for accuracy
#                 if np.argmax(output) == np.argmax(target):
#                     correct_predictions += 1
#                 print(f"correct_predictions: {correct_predictions}")

#             # Calculate average loss and accuracy for the epoch
#             avg_loss = total_loss / len(X)
#             accuracy = correct_predictions / len(X) * 100
#             # Append to the lists for visualization
#             self.train_losses.append(avg_loss)
#             self.train_accuracies.append(accuracy)

#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

#         print("Training complete.")

#     def predict(self, X):
#         print("Predicting")
#         predictions = []
#         total_images = len(X)
#         for index, x in enumerate(X):
#             output = self.forward(x)
#             predictions.append(output.argmax())
#             print(f"Prediction: {index + 1}/{total_images}", end='\r')
#         return predictions

# # Load and preprocess the images
# def load_and_preprocess_images(image_paths):
#     print("Loading and Preprocessing Data")
#     images = []
#     for index, image_path in enumerate(image_paths):
#         # Load the image and preprocess it (resize and normalize)
#         image = Image.open(image_path).convert('L')  # Convert to grayscale
#         image = image.resize((32, 32))  # Resize to a consistent size
#         image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

#         # Apply Sobel edge detection manually using nested loops
#         gradient_x = np.zeros_like(image, dtype=np.float64)
#         gradient_y = np.zeros_like(image, dtype=np.float64)
#         for i in range(1, image.shape[0] - 1):
#             for j in range(1, image.shape[1] - 1):
#                 gradient_x[i, j] = (
#                     sobel_x[0, 0] * image[i - 1, j - 1] +
#                     sobel_x[0, 1] * image[i - 1, j] +
#                     sobel_x[0, 2] * image[i - 1, j + 1] +
#                     sobel_x[1, 0] * image[i, j - 1] +
#                     sobel_x[1, 1] * image[i, j] +
#                     sobel_x[1, 2] * image[i, j + 1] +
#                     sobel_x[2, 0] * image[i + 1, j - 1] +
#                     sobel_x[2, 1] * image[i + 1, j] +
#                     sobel_x[2, 2] * image[i + 1, j + 1]
#                 )

#                 gradient_y[i, j] = (
#                     sobel_y[0, 0] * image[i - 1, j - 1] +
#                     sobel_y[0, 1] * image[i - 1, j] +
#                     sobel_y[0, 2] * image[i - 1, j + 1] +
#                     sobel_y[1, 0] * image[i, j - 1] +
#                     sobel_y[1, 1] * image[i, j] +
#                     sobel_y[1, 2] * image[i, j + 1] +
#                     sobel_y[2, 0] * image[i + 1, j - 1] +
#                     sobel_y[2, 1] * image[i + 1, j] +
#                     sobel_y[2, 2] * image[i + 1, j + 1]
#                 )

#         edge_image = np.sqrt(gradient_x**2 + gradient_y**2)
#         images.append(edge_image)
#         print(f"Preprocessing image {index + 1}/{len(image_paths)}", end='\r')
#     return np.array(images)


# # Create and train the neural network
# input_size = 32 * 32  # Assuming 32x32 edge images
# hidden_size = 128  # Choose an appropriate size
# num_classes = 10  # 10 classes (digits 0 to 9)
# output_size = num_classes
# learning_rate = 0.001
# epochs = 20  # Adjust the number of epochs as needed

# # Training phase
# print("Training the neural network...")
# model = NeuralNetwork(input_size, hidden_size, output_size)
# model.train(X_train, y_train, epochs, learning_rate)
# print("Training complete.")

# # Validation phase (optional)
# print("Validation phase...")
# validation_predictions = model.predict(X_val)
# total_correct = np.sum(np.array(validation_predictions) == np.argmax(y_val))
# validation_accuracy = total_correct / len(X_val)
# print(f"Validation Accuracy: {validation_accuracy:.2%}")
# # Print the first few predictions and true labels for investigation
# num_samples_to_print = 10  # Adjust the number of samples to print as needed

# print("True Labels:", y_val[:num_samples_to_print])
# print("Predictions:", validation_predictions[:num_samples_to_print])

# print("Validation complete.")

# # Prediction phase
# print("Making predictions on the test set...")
# predictions = model.predict(X_test)
# correct_predictions = np.sum(np.array(validation_predictions) == np.array(y_val))
# print(predictions)
# print(correct_predictions)
# print("Predictions complete.")

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











# # # Create and train the neural network
# # input_size = 32 * 32  # Assuming 32x32 edge images
# # hidden_size = 128  # Choose an appropriate size
# # output_size = 10  # 10 classes (digits 0 to 9)
# # learning_rate = 0.001
# # epochs = 20  # Adjust the number of epochs as needed

# # # Training phase
# # print("Training the neural network...")
# # model = NeuralNetwork(input_size, hidden_size, output_size)
# # model.train(X_train, y_train, epochs, learning_rate)
# # print("Training complete.")

# # # Validation phase (optional)
# # print("Validation phase...")
# # validation_predictions = model.predict(X_val)
# # total_correct = np.sum(np.array(validation_predictions) == y_val)
# # validation_accuracy = total_correct / len(X_val)
# # print(f"Validation Accuracy: {validation_accuracy:.2%}")
# # # Print the first few predictions and true labels for investigation
# # num_samples_to_print = 10  # Adjust the number of samples to print as needed

# # print("True Labels:", y_val[:num_samples_to_print])
# # print("Predictions:", validation_predictions[:num_samples_to_print])
# # print("Validation complete.")

# # # Prediction phase
# # print("Making predictions on the test set...")
# # predictions = model.predict(X_test)
# # correct_predictions = np.sum(np.array(predictions) == y_test)
# # print(f"Test Accuracy: {correct_predictions / len(X_test) * 100:.2f}%")
# # print(predictions)
# # print("Predictions complete.")

# # # You can now use the 'predictions' variable for further analysis or evaluation









# # # Import necessary libraries
# # import os
# # import numpy as np
# # from PIL import Image
# # from sklearn.model_selection import train_test_split
# # import shutil
# # import matplotlib.pyplot as plt

# # # Define the source directory where the digit folders are located
# # source_dir = './numerals'  # Replace with the path to your images folder

# # # Define the destination directories for train, validation, and test splits
# # train_dir = './numerals_train'
# # validation_dir = './numerals_validation'
# # test_dir = './numerals_test'

# # # Create destination directories if they don't exist
# # for directory in [train_dir, validation_dir, test_dir]:
# #     if not os.path.exists(directory):
# #         os.makedirs(directory)

# # # Initialize counters for each class
# # class_counts = {str(i): 0 for i in range(10)}

# # # Iterate through each digit folder (0 to 9)
# # for digit in range(10):
# #     digit_folder = os.path.join(source_dir, str(digit))
    
# #     # Get a list of image files in the folder
# #     image_files = [filename for filename in os.listdir(digit_folder) if filename.endswith('.jpg')]
    
# #     # Split the image files into train, validation, and test sets
# #     train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
# #     val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
# #     # Copy images to the respective directories with modified names
# #     for filename in train_files:
# #         src_path = os.path.join(digit_folder, filename)
# #         dst_filename = f"{digit}_{filename}"  # Modified filename
# #         dst_path = os.path.join(train_dir, dst_filename)
# #         shutil.copy(src_path, dst_path)
# #         class_counts[str(digit)] += 1
    
# #     for filename in val_files:
# #         src_path = os.path.join(digit_folder, filename)
# #         dst_filename = f"{digit}_{filename}"  # Modified filename
# #         dst_path = os.path.join(validation_dir, dst_filename)
# #         shutil.copy(src_path, dst_path)
# #         class_counts[str(digit)] += 1
    
# #     for filename in test_files:
# #         src_path = os.path.join(digit_folder, filename)
# #         dst_filename = f"{digit}_{filename}"  # Modified filename
# #         dst_path = os.path.join(test_dir, dst_filename)
# #         shutil.copy(src_path, dst_path)
# #         class_counts[str(digit)] += 1


# # # Print the number of images in each class in the train, validation, and test splits
# # for digit in range(10):
# #     print(f"Class {digit}: {class_counts[str(digit)]} images")

# # # Define the sigmoid activation function and its derivative
# # def sigmoid(x):
# #     return 1 / (1 + np.exp(-x))

# # def sigmoid_derivative(x):
# #     return x * (1 - x)

# # # Define the Sobel operator kernels for gradient calculation
# # sobel_x = np.array([[-1, 0, 1],
# #                     [-2, 0, 2],
# #                     [-1, 0, 1]])

# # sobel_y = np.array([[-1, -2, -1],
# #                     [0, 0, 0],
# #                     [1, 2, 1]])

# # # Define the neural network class
# # class NeuralNetwork:
# #     def __init__(self, input_size, hidden_size, output_size):
# #         # Initialize the weights and biases with random values
# #         print("Initializing NN")
# #         self.weights_input_hidden = np.random.rand(input_size, hidden_size)
# #         self.bias_hidden = np.zeros((1, hidden_size))
# #         self.weights_hidden_output = np.random.rand(hidden_size, output_size)
# #         self.bias_output = np.zeros((1, output_size))
# #         self.train_losses = []  # List to store training loss values
# #         self.val_losses = []    # List to store validation loss values
# #         self.train_accuracies = []  # List to store training accuracy values
# #         self.val_accuracies = []    # List to store validation accuracy values

# #     def forward(self, x):
# #         # Flatten the input data to match the input layer's weights
# #         x = x.reshape(1, -1)

# #         # Calculate the output of the hidden layer
# #         self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
# #         self.hidden_output = sigmoid(self.hidden_input)

# #         # Calculate the final output
# #         self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
# #         self.final_output = sigmoid(self.final_input)

# #         return self.final_output

# #     def backward(self, x, y, learning_rate):
# #         # Calculate the loss
# #         loss = y - self.final_output

# #         # Calculate the gradients and update the weights and biases
# #         output_delta = loss * sigmoid_derivative(self.final_output)
# #         hidden_loss = output_delta.dot(self.weights_hidden_output.T)
# #         hidden_delta = hidden_loss * sigmoid_derivative(self.hidden_output)

# #         self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
# #         self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
# #         self.weights_input_hidden += x.reshape(-1, 1).dot(hidden_delta.reshape(1, -1)) * learning_rate
# #         self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# #     def train(self, X, y, epochs, learning_rate):
# #         print("Training")
# #         # Training phase
# #         print("Training the neural network...")
# #         for epoch in range(epochs):
# #             total_loss = 0
# #             correct_predictions = 0

# #             for i in range(len(X)):
# #                 x = X[i]
# #                 target = y[i]

# #                 # Forward pass
# #                 output = model.forward(x)

# #                 # Compute loss
# #                 loss = np.mean(0.5 * (target - output) ** 2)
# #                 total_loss += loss

# #                 # Backward pass
# #                 model.backward(x, target, learning_rate)

# #                 # Count correct predictions for accuracy
# #                 if np.argmax(output) == np.argmax(target):
# #                     correct_predictions += 1

# #             # Calculate average loss and accuracy for the epoch
# #             avg_loss = total_loss / len(X)
# #             accuracy = correct_predictions / len(X) * 100

# #             # Append to the lists for visualization
# #             self.train_losses.append(avg_loss)
# #             self.train_accuracies.append(accuracy)

# #             print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# #         print("Training complete.")

# #     def predict(self, X):
# #         print("Predicting")
# #         predictions = []
# #         total_images = len(X)
# #         for index, x in enumerate(X):
# #             output = self.forward(x)
# #             predictions.append(output.argmax())
# #             print(f"Prediction: {index + 1}/{total_images}", end='\r')
# #         return predictions

# # # Load and preprocess the images
# # def load_and_preprocess_images(image_paths):
# #     print("Loading and Preprocessing Data")
# #     images = []
# #     for index, image_path in enumerate(image_paths):
# #         # Load the image and preprocess it (resize and normalize)
# #         image = Image.open(image_path).convert('L')  # Convert to grayscale
# #         image = image.resize((32, 32))  # Resize to a consistent size
# #         image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

# #         # Apply Sobel edge detection manually using nested loops
# #         gradient_x = np.zeros_like(image, dtype=np.float64)
# #         gradient_y = np.zeros_like(image, dtype=np.float64)
# #         for i in range(1, image.shape[0] - 1):
# #             for j in range(1, image.shape[1] - 1):
# #                 gradient_x[i, j] = (
# #                     sobel_x[0, 0] * image[i - 1, j - 1] +
# #                     sobel_x[0, 1] * image[i - 1, j] +
# #                     sobel_x[0, 2] * image[i - 1, j + 1] +
# #                     sobel_x[1, 0] * image[i, j - 1] +
# #                     sobel_x[1, 1] * image[i, j] +
# #                     sobel_x[1, 2] * image[i, j + 1] +
# #                     sobel_x[2, 0] * image[i + 1, j - 1] +
# #                     sobel_x[2, 1] * image[i + 1, j] +
# #                     sobel_x[2, 2] * image[i + 1, j + 1]
# #                 )

# #                 gradient_y[i, j] = (
# #                     sobel_y[0, 0] * image[i - 1, j - 1] +
# #                     sobel_y[0, 1] * image[i - 1, j] +
# #                     sobel_y[0, 2] * image[i - 1, j + 1] +
# #                     sobel_y[1, 0] * image[i, j - 1] +
# #                     sobel_y[1, 1] * image[i, j] +
# #                     sobel_y[1, 2] * image[i, j + 1] +
# #                     sobel_y[2, 0] * image[i + 1, j - 1] +
# #                     sobel_y[2, 1] * image[i + 1, j] +
# #                     sobel_y[2, 2] * image[i + 1, j + 1]
# #                 )

# #         edge_image = np.sqrt(gradient_x**2 + gradient_y**2)
# #         images.append(edge_image)
# #         print(f"Preprocessing image {index + 1}/{len(image_paths)}", end='\r')
# #     return np.array(images)

# # # Replace with your image file paths and corresponding labels
# # image_paths = []  # List of file paths to your images
# # labels = []       # Corresponding labels (0 to 9) for each image
# # # Iterate through each digit folder (0 to 9)
# # for digit in range(10):
# #     digit_folder = os.path.join(source_dir, str(digit))
    
# #     # Get a list of image files in the folder
# #     image_files = [filename for filename in os.listdir(digit_folder) if filename.endswith('.jpg')]
    
# #     # Append the image paths and corresponding labels to the lists
# #     image_paths.extend([os.path.join(digit_folder, filename) for filename in image_files])
# #     labels.extend([digit] * len(image_files))

# # # # Split the data into train, validation, and test sets
# # # X_train, X_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.3, random_state=42)
# # # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



# # # Assuming X_train is a list of file paths
# # for filename in X_train:
# #     label = os.path.basename(filename)[0]
# #     print(f"TRAIN: Filename: {filename}, First Character: {label}")

# # # Assuming X_train is a list of file paths
# # for filename in X_val:
# #     label = os.path.basename(filename)[0]
# #     print(f"VAL: Filename: {filename}, First Character: {label}")


# # # Assuming X_train is a list of file paths
# # for filename in X_test:
# #     label = os.path.basename(filename)[0]
# #     print(f"TEST: Filename: {filename}, First Character: {label}")



# # # y_train = [int(os.path.basename(filename)[0]) for filename in X_train]
# # # y_val = [int(os.path.basename(filename)[0]) for filename in X_val]
# # # y_test = [int(os.path.basename(filename)[0]) for filename in X_test]

# # # # Load and preprocess the images
# # # X_train = load_and_preprocess_images(X_train)
# # # X_val = load_and_preprocess_images(X_val)
# # # X_test = load_and_preprocess_images(X_test)

# # # # Convert labels to one-hot encoding (assuming label values are integers)
# # # num_classes = 10  # 10 classes (digits 0 to 9)
# # # # y_train = np.eye(num_classes)[y_train]
# # # # y_val = np.eye(num_classes)[y_val]
# # # # y_test = np.eye(num_classes)[y_test]

# # # # # Replace the previous labels with labels based on the first digit of the file name for training
# # # # y_train = [int(filename[0]) for filename in train_files]
# # # # y_val = [int(filename[0]) for filename in val_files]

# # # # Replace the previous labels with labels based on the first digit of the file name for training



# # # # Create and train the neural network
# # # input_size = 32 * 32  # Assuming 32x32 edge images
# # # hidden_size = 128  # Choose an appropriate size
# # # output_size = num_classes
# # # learning_rate = 0.001
# # # epochs = 20  # Adjust the number of epochs as needed

# # # # Training phase
# # # print("Training the neural network...")
# # # model = NeuralNetwork(input_size, hidden_size, output_size)
# # # model.train(X_train, y_train, epochs, learning_rate)
# # # print("Training complete.")

# # # # Validation phase (optional)
# # # print("Validation phase...")
# # # validation_predictions = model.predict(X_val)
# # # total_correct = np.sum(np.array(validation_predictions) == np.argmax(y_val))
# # # validation_accuracy = total_correct / len(X_val)
# # # print(f"Validation Accuracy: {validation_accuracy:.2%}")
# # # # Print the first few predictions and true labels for investigation
# # # num_samples_to_print = 10  # Adjust the number of samples to print as needed

# # # print("True Labels:", y_val[:num_samples_to_print])
# # # print("Predictions:", validation_predictions[:num_samples_to_print])

# # # print("Validation complete.")

# # # # Prediction phase
# # # print("Making predictions on the test set...")
# # # predictions = model.predict(X_test)
# # # correct_predictions = np.sum(np.array(validation_predictions) == np.array(y_val))
# # # print(predictions)
# # # print(correct_predictions)
# # # print("Predictions complete.")

# # # import random

# # # # Function to display images and predictions
# # # def display_images_with_predictions(images, true_labels, predictions, class_names):
# # #     num_samples = len(images)
# # #     sample_indices = random.sample(range(num_samples), 3)  # Select 3 random samples
    
# # #     plt.figure(figsize=(12, 4))
    
# # #     for i, index in enumerate(sample_indices):
# # #         plt.subplot(1, 3, i + 1)
# # #         plt.imshow(images[index], cmap='gray')
# # #         plt.title(f"True: {class_names[true_labels[index]]}\nPredicted: {class_names[predictions[index]]}")
# # #         plt.axis('off')

# # # # List of class names (replace with your class names if needed)
# # # class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"]

# # # # Display images with predictions
# # # display_images_with_predictions(X_test, y_test, predictions, class_names)
# # # plt.show()


# # # # Plot the learning curves (loss and accuracy)
# # # plt.figure(figsize=(12, 4))
# # # plt.subplot(1, 2, 1)
# # # plt.plot(model.train_losses, label='Training Loss')
# # # plt.plot(model.val_losses, label='Validation Loss')
# # # plt.xlabel('Epoch')
# # # plt.ylabel('Loss')
# # # plt.legend()
# # # plt.title('Training and Validation Loss')

# # # plt.subplot(1, 2, 2)
# # # plt.plot(model.train_accuracies, label='Training Accuracy')
# # # plt.plot(model.val_accuracies, label='Validation Accuracy')
# # # plt.xlabel('Epoch')
# # # plt.ylabel('Accuracy (%)')
# # # plt.legend()
# # # plt.title('Training and Validation Accuracy')

# # # plt.tight_layout()
# # # plt.show()

# # # # You can now use the 'predictions' variable for further analysis or evaluation
