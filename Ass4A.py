#importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step b: Upload / Access the Dataset
data = pd.read_csv('creditcard.csv')  # Ensure that 'creditcard.csv' is in the current directory or provide the correct path

# Preprocess the data
X = data.drop(columns=['Class'])  # Features (input)
y = data['Class']  # Labels (output)


# Normalize the features (Scaling the input data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step c: Encoder - Convert input data to a latent representation
input_dim = X_train.shape[1]  # Number of features
input_layer = Input(shape=(input_dim,))
# Encoder network
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)  # Latent representation

# Step d: Decoder - Reconstruct the input data from the latent representation
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)  # Reconstructed output layer, should match the input dimensions


# Step e: Combine encoder and decoder to create the full Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile the model with optimizer, loss function, and evaluation metrics
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Summary of the model architecture
autoencoder.summary()


# Train the Autoencoder model
history = autoencoder.fit(
    X_train, X_train,  # We are training the model to reconstruct the input data (X_train)
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_data=(X_test, X_test)  # Using X_test for validation
)


# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Step: Anomaly Detection using Reconstruction Error
# Use the trained autoencoder model to predict on the test set
X_test_pred = autoencoder.predict(X_test)

# Calculate the reconstruction error (MSE) for each data point
reconstruction_error = np.mean(np.square(X_test - X_test_pred), axis=1)

# Define a threshold for anomaly detection (95th percentile of the reconstruction error)
threshold = np.percentile(reconstruction_error, 95)

# Anomalies are those data points where the reconstruction error is greater than the threshold
anomalies = reconstruction_error > threshold


# Step: Evaluation using Confusion Matrix and Classification Report
# Evaluate the performance of the anomaly detection model by comparing predicted anomalies with true labels
print("Confusion Matrix:\n", confusion_matrix(y_test, anomalies))
print("\nClassification Report:\n", classification_report(y_test, anomalies))

