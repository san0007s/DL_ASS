import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# Define path to data folder containing subfolders for each class
data_folder = 'caltech-101-img'  # Replace with the path to your main data folder

# Define data generator with validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% of data will be used for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training data (80%) and validation data (20%) from the same directory
train_data = datagen.flow_from_directory(
    data_folder,
    target_size=(32, 32),  # Change target size to (32, 32)
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_folder,
    target_size=(32, 32),  # Change target size to (32, 32)
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Automatically set num_classes based on the number of subfolders (classes)
num_classes = len(train_data.class_indices)

# Load pre-trained VGG16 model
base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(32, 32, 3))  # Adjusted to (32, 32, 3)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Train only the custom classifier layers
model.fit(train_data, steps_per_epoch=20, epochs=10, validation_data=val_data)

# Fine-tune by unfreezing some layers of the base model
for layer in base_model.layers[-10:]:  # Adjust the number of layers to unfreeze as needed
    layer.trainable = True

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
model.fit(train_data, epochs=1, validation_data=val_data)

# List class labels
class_labels = list(train_data.class_indices.keys())

# Simple prediction and visualization using val_data
x_val, y_val = val_data.next()  # Load a batch of validation data
pred = model.predict(x_val)  # Make predictions

# Display one of the images along with its predicted and true label
index = 10  # Change the index to see a different image
plt.imshow(x_val[index])  # Display the image
plt.title(f"Predicted: {class_labels[np.argmax(pred[index])]}, True: {class_labels[np.argmax(y_val[index])]}")
plt.axis('off')
plt.show()
