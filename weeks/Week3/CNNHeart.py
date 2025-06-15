import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load Data
normal_df = pd.read_csv("ptbdb_normal.csv").iloc[:, :-1]
anomaly_df = pd.read_csv("ptbdb_abnormal.csv").iloc[:, :-1]

# Data Preprocessing
normal_df_copy = normal_df.copy()
anomaly_df_copy = anomaly_df.copy()
normal_df_copy = normal_df_copy.set_axis(range(1, 188), axis=1)
anomaly_df_copy = anomaly_df_copy.set_axis(range(1, 188), axis=1)
normal_df_copy = normal_df_copy.assign(target="Normal")
anomaly_df_copy = anomaly_df_copy.assign(target="Anomaly")

# Concatenate Data
df = pd.concat((normal_df_copy, anomaly_df_copy))

# Prepare Data for Training (Only Normal and Anomaly)
normal = normal_df.to_numpy()
anomaly = anomaly_df.to_numpy()
X_train, X_test = train_test_split(normal, test_size=0.15, random_state=45, shuffle=True)

# CNN Model for Outlier Detection
class AutoEncoder(keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Reshape((input_dim, 1)),
            layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(latent_dim, 3, strides=1, activation='relu', padding="same"),
            layers.MaxPooling1D(2, padding="same"),
        ])

        self.decoder = keras.Sequential([
            layers.Conv1DTranspose(latent_dim, 3, strides=1, activation='relu', padding="same"),
            layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
            layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
            layers.Flatten(),
            layers.Dense(input_dim)
        ])

    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded


# Model Training
input_dim = X_train.shape[-1]
latent_dim = 32
model = AutoEncoder(input_dim, latent_dim)
model.build((None, input_dim))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mae")

# Train the Model (This will train the model just once)
history = model.fit(X_train, X_train, epochs=100, batch_size=128, validation_split=0.1)

# Model Evaluation
train_mae = model.evaluate(X_train, X_train, verbose=0)
test_mae = model.evaluate(X_test, X_test, verbose=0)
anomaly_mae = model.evaluate(anomaly_df, anomaly_df, verbose=0)

print("Training dataset error: ", train_mae)
print("Testing dataset error: ", test_mae)
print("Anomaly dataset error: ", anomaly_mae)

# Anomaly Detection: Use MAE to Detect Outliers
threshold = np.mean(train_mae) + 3 * np.std(train_mae)
predictions = model.predict(X_test)
mae = np.mean(np.abs(predictions - X_test), axis=1)

# Identify Outliers
outliers = np.where(mae > threshold)
print(f"Detected Outliers: {len(outliers[0])} samples")

# Get Example of Normal Sample (No re-training)
normal_sample_idx = 0  # Example index of normal sample
normal_sample = X_test[normal_sample_idx]
normal_pred = model.predict(np.expand_dims(normal_sample, axis=0))

print("\nNormal Sample:")
print(f"True: {normal_sample[:10]}...")  # Show first 10 values of normal sample
print(f"Predicted: {normal_pred[0][:10]}...")  # Show first 10 predicted values

# Calculate MAE for Normal Sample
normal_sample_mae = mean_absolute_error(normal_sample, normal_pred[0])
print(f"Normal Sample MAE: {normal_sample_mae}")

# Get Example of Test Sample (Prediction on a test sample)
test_sample_idx = 0  # Example index of test sample from X_test
test_sample = X_test[test_sample_idx]
test_pred = model.predict(np.expand_dims(test_sample, axis=0))

print("\nTest Sample:")
print(f"True: {test_sample[:10]}...")  # Show first 10 values of test sample
print(f"Predicted: {test_pred[0][:10]}...")  # Show first 10 predicted values

# Calculate MAE for Test Sample
test_sample_mae = mean_absolute_error(test_sample, test_pred[0])
print(f"Test Sample MAE: {test_sample_mae}")

# Get Example of Anomaly Sample (No re-training)
anomaly_sample_idx = 0  # Example index of anomaly sample
anomaly_sample = anomaly[anomaly_sample_idx]
anomaly_pred = model.predict(np.expand_dims(anomaly_sample, axis=0))

print("\nAnomaly Sample:")
print(f"True: {anomaly_sample[:10]}...")  # Show first 10 values of anomaly sample
print(f"Predicted: {anomaly_pred[0][:10]}...")  # Show first 10 predicted values

# Calculate MAE for Anomaly Sample
anomaly_sample_mae = mean_absolute_error(anomaly_sample, anomaly_pred[0])
print(f"Anomaly Sample MAE: {anomaly_sample_mae}")






