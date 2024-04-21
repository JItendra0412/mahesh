import os  # For file system operations
import cv2  # For image processing
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import LabelEncoder  # For encoding categorical labels
from tensorflow.keras import layers, models  # For building the Convolutional Neural Network (CNN) model
from sklearn.metrics import classification_report  # For generating a classification report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []  # Initialize an empty list to store images
    labels = []  # Initialize an empty list to store labels

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)  # Construct the full image path

        # Check if the file is a valid image format (JPEG)
        if img_path.endswith(".jpeg") or img_path.endswith(".jpg"):
            # Read the image using OpenCV
            img = cv2.imread(img_path)

            # Resize the image to a common size if needed
            img = cv2.resize(img, (224, 224))  # Adjust size as needed

            # Normalize pixel values to be between 0 and 1
            img = img / 255.0

            # Append the image and label to the lists
            images.append(img)
            labels.append(filename.split('.')[0])  # Assuming filename contains the label

    # Convert lists to NumPy arrays for further processing
    return np.array(images), np.array(labels)

# Specify the paths to your training and testing image folders
train_folder = "/content/sample_data/normal retinoblasma"
test_folder = "/content/sample_data/retino"

# Load images and labels for training and testing datasets
X_train, y_train = load_images_from_folder(train_folder)
X_test, y_test = load_images_from_folder(test_folder)

# Encode labels using LabelEncoder for training and testing datasets combined
label_encoder = LabelEncoder()
y_combined = np.concatenate((y_train, y_test), axis=0)
y_combined_encoded = label_encoder.fit_transform(y_combined)

# Split the combined encoded labels back into training and testing sets
y_train_encoded = y_combined_encoded[:len(y_train)]
y_test_encoded = y_combined_encoded[len(y_train):]

# Define the number of unique classes in your dataset
num_classes = len(np.unique(y_combined))

# Convert integer labels to one-hot encoded labels
y_train_encoded = to_categorical(y_train_encoded, num_classes)
y_test_encoded = to_categorical(y_test_encoded, num_classes)

# Create an ImageDataGenerator instance for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Fit the data generator on the training data
datagen.fit(X_train)

# Define the output layer with the correct number of units
output_units = num_classes

# Define the CNN model architecture with Leaky ReLU activation
model = models.Sequential([
    # Add a 2D convolutional layer with 32 filters, a 3x3 kernel, and Leaky ReLU activation function
    layers.Conv2D(32, (3, 3), activation=layers.LeakyReLU(alpha=0.1), input_shape=(224, 224, 3)),

    # Add a 2D max pooling layer with a 2x2 pool size
    layers.MaxPooling2D((2, 2)),

    # Add a 2D convolutional layer with 64 filters, a 3x3 kernel, and Leaky ReLU activation function
    layers.Conv2D(64, (3, 3), activation=layers.LeakyReLU(alpha=0.1)),

    # Add a 2D max pooling layer with a 2x2 pool size
    layers.MaxPooling2D((2, 2)),

    # Add a 2D convolutional layer with 128 filters, a 3x3 kernel, and Leaky ReLU activation function
    layers.Conv2D(128, (3, 3), activation=layers.LeakyReLU(alpha=0.1)),

    # Add a 2D max pooling layer with a 2x2 pool size
    layers.MaxPooling2D((2, 2)),

    # Flatten the output of the convolutional layers to a 1D array
    layers.Flatten(),

    # Add a dense (fully connected) layer with 128 units and Leaky ReLU activation function
    layers.Dense(128, activation=layers.LeakyReLU(alpha=0.1)),

    # Add the output layer with the number of units equal to the number of classes and softmax activation function
    layers.Dense(output_units, activation='softmax')
])

# Compile the model by specifying the optimizer, loss function, and evaluation metric
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using the fit method with data augmentation
history = model.fit(datagen.flow(X_train, y_train_encoded, batch_size=32),
                    epochs=10,
                    validation_data=(X_test, y_test_encoded))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print('Test accuracy:', test_acc)

# Predict on the test set
def predict_on_test_set(model, X_test):
    y_pred_encoded = np.argmax(model.predict(X_test), axis=-1)
    return y_pred_encoded

y_pred_encoded = predict_on_test_set(model, X_test)

# Decode one-hot encoded labels back to integer labels
y_test_decoded = np.argmax(y_test_encoded, axis=1)
y_pred_decoded = y_pred_encoded

# Print a classification report to evaluate the model performance
print(classification_report(y_test_decoded, y_pred_decoded, labels=np.unique(y_test_decoded), zero_division=1))
main code
