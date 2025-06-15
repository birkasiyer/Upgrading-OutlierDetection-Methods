import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Input
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# 1. Load and prepare the data
data = load_breast_cancer()
X, y = data.data, data.target  # X: Features, y: Labels (0: malignant, 1: benign)

# Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 2. Outlier detection with OCSVM
ocsvm = OneClassSVM(nu=0.05, kernel='rbf')
ocsvm_outliers = ocsvm.fit_predict(X_pca)

# Create normal and abnormal classes from OCSVM
ocsvm_labels = np.where(ocsvm_outliers == 1, 1, 0)

# 3. Outlier detection with IQR
def detect_outliers_iqr(data):
    outliers = np.zeros(data.shape[0], dtype=bool)
    for col in range(data.shape[1]):
        Q1 = np.percentile(data[:, col], 25)
        Q3 = np.percentile(data[:, col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers |= (data[:, col] < lower_bound) | (data[:, col] > upper_bound)
    return outliers

iqr_outliers = detect_outliers_iqr(X_scaled)

# 4. Outlier detection with Z-Score
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    return np.any(z_scores > threshold, axis=1)

zscore_outliers = detect_outliers_zscore(X_scaled)

# Combine labels
combined_labels = np.where(
    (ocsvm_labels == 0) | (iqr_outliers == True) | (zscore_outliers == True), 0, 1
)

# Count the outliers detected by each method
ocsvm_indices = np.where(ocsvm_labels == 0)[0]
iqr_indices = np.where(iqr_outliers)[0]
zscore_indices = np.where(zscore_outliers)[0]

# Find unique outlier indices
all_outlier_indices = np.unique(np.concatenate([ocsvm_indices, iqr_indices, zscore_indices]))
total_outliers = len(all_outlier_indices)

# Find common outliers detected by all three methods
common_outliers = np.intersect1d(np.intersect1d(ocsvm_indices, iqr_indices), zscore_indices)
common_outlier_count = len(common_outliers)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, combined_labels, test_size=0.3, random_state=42)

# 5. CNN Model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=1)

# Model evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
predictions = (model.predict(X_test) > 0.5).astype(int)

# Performance metrics
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Classification report
report = classification_report(y_test, predictions, target_names=["Outlier", "Normal"])
print("=== Classification Report ===")
print(report)

# Visualization
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Scatter plot
ax[0].scatter(X_pca[combined_labels == 1, 0], X_pca[combined_labels == 1, 1],
              c='blue', s=10, label=f'Normal')
ax[0].scatter(X_pca[ocsvm_indices, 0], X_pca[ocsvm_indices, 1],
              c='orange', s=10, label=f'OCSVM Outliers ({len(ocsvm_indices)})')
ax[0].scatter(X_pca[iqr_indices, 0], X_pca[iqr_indices, 1],
              c='green', s=10, label=f'IQR Outliers ({len(iqr_indices)})')
ax[0].scatter(X_pca[zscore_indices, 0], X_pca[zscore_indices, 1],
              c='purple', s=10, label=f'Z-Score Outliers ({len(zscore_indices)})')
ax[0].scatter(X_pca[common_outliers, 0], X_pca[common_outliers, 1],
              c='red', s=10, label=f'Common Outliers ({common_outlier_count})')

# Title and legend
ax[0].set_title("Combined OCSVM + IQR + Z-Score Results (PCA Reduced Data)", fontsize=14)
ax[0].set_xlabel("PCA1")
ax[0].set_ylabel("PCA2")
ax[0].legend(loc='upper right')
ax[0].grid(True)

# Text summary
text_str = (
    f"=== Total Outlier Detection Results ===\n"
    f"OCSVM Outliers: {len(ocsvm_indices)}\n"
    f"IQR Outliers: {len(iqr_indices)}\n"
    f"Z-Score Outliers: {len(zscore_indices)}\n"
    f"Common Outliers: {common_outlier_count}\n"
    f"Total Unique Outliers: {total_outliers}\n\n"
    f"=== CNN Performance Metrics ===\n"
    f"Precision: {precision * 100:.2f}%\n"
    f"Recall: {recall * 100:.2f}%\n"
    f"F1-Score: {f1 * 100:.2f}%"
)
ax[1].text(0.5, 0.8, text_str, wrap=True, horizontalalignment='center', fontsize=12, verticalalignment='bottom')
ax[1].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

# Print results in terminal
print("=== Total Outlier Detection Results ===")
print(f"OCSVM Outliers: {len(ocsvm_indices)}")
print(f"IQR Outliers: {len(iqr_indices)}")
print(f"Z-Score Outliers: {len(zscore_indices)}")
print(f"Common Outliers: {common_outlier_count}")
print(f"Total Unique Outliers: {total_outliers}\n")

print("=== CNN Performance Metrics ===")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")