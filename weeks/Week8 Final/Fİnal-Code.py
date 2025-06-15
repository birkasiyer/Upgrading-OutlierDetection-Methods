import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Input
from keras.api.optimizers import Adam
import keras_tuner as kt
import warnings

# Dataset and StandartScaler
digits = load_digits()
X, y = digits.data, digits.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# OCSVM Parameters
ocsvm_per_class_outliers = {}
ocsvm_total_outliers = 0
ocsvm_labels_all = np.zeros_like(y)
y_ocsvm_eval = np.ones_like(y)
param_grid = {'nu': [0.23], 'kernel': ['rbf']}

# OCSVM
def ocswm_score_func(estimator, X):
    return np.mean(estimator.predict(X) == 1)

ocsvm_grid = GridSearchCV(OneClassSVM(), param_grid, cv=3, scoring=ocswm_score_func)
ocsvm_grid.fit(X_scaled)
best_ocsvm = ocsvm_grid.best_estimator_

# One-Class SVM
for i in range(10):
    class_indices = np.where(y == i)[0]
    X_class = X_scaled[class_indices]
    ocsvm_outliers = best_ocsvm.fit_predict(X_class)
    ocsvm_labels = np.where(ocsvm_outliers == 1, 1, 0)
    ocsvm_labels_all[class_indices] = ocsvm_labels
    y_ocsvm_eval[class_indices[ocsvm_labels == 0]] = 0
    ocsvm_per_class_outliers[i] = np.sum(ocsvm_labels == 0)
    ocsvm_total_outliers += np.sum(ocsvm_labels == 0)

# LOF - Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20)
lof_outliers = lof.fit_predict(X_scaled)
lof_labels = np.where(lof_outliers == -1, 0, 1)

# Isolation Forest
iforest = IsolationForest(contamination=0.1)
iforest_outliers = iforest.fit_predict(X_scaled)
iforest_labels = np.where(iforest_outliers == -1, 0, 1)

# Elliptic Envelope
envelope = EllipticEnvelope(contamination=0.1)
warnings.filterwarnings("ignore", message="The covariance matrix associated to your dataset is not full rank")
envelope_outliers = envelope.fit_predict(X_scaled)
envelope_labels = np.where(envelope_outliers == -1, 0, 1)

# CNN
y_with_outliers = np.where(ocsvm_labels_all == 0, 10, y)

all_outliers = np.union1d(np.union1d(np.union1d(np.where(ocsvm_labels_all == 0)[0],
                                                np.where(lof_labels == 0)[0]),
                                     np.where(iforest_labels == 0)[0]),
                          np.where(envelope_labels == 0)[0])

y_with_outliers[all_outliers] = 10

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_with_outliers, test_size=0.4, random_state=42)

X_train_combined = np.hstack([X_train, lof_labels[:len(X_train)].reshape(-1, 1),
                              iforest_labels[:len(X_train)].reshape(-1, 1),
                              envelope_labels[:len(X_train)].reshape(-1, 1)])

X_test_combined = np.hstack([X_test, lof_labels[:len(X_test)].reshape(-1, 1),
                             iforest_labels[:len(X_test)].reshape(-1, 1),
                             envelope_labels[:len(X_test)].reshape(-1, 1)])

# CNN Model
class CNNHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(X_train_combined.shape[1],)))
        model.add(Dense(units=hp.Int('units_1', min_value=64, max_value=256, step=64), activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu'))
        model.add(Dense(11, activation='softmax'))  # 10 classes (0-9) + 1 for outliers (10)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

tuner = kt.RandomSearch(
    CNNHyperModel(),
    objective=kt.Objective("val_accuracy", direction="max"),
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='digit_classifier',
    overwrite=True
)

tuner.search(X_train_combined, y_train, epochs=20, batch_size=32, validation_data=(X_test_combined, y_test), verbose=1)
best_cnn_model = tuner.get_best_models(num_models=1)[0]
best_cnn_model.optimizer = None
best_cnn_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# CNN Test
test_loss, test_accuracy = best_cnn_model.evaluate(X_test_combined, y_test, verbose=1)
predictions = np.argmax(best_cnn_model.predict(X_test_combined), axis=1)
predictions_with_outliers = np.where(predictions == 10, 10, predictions)

cnn_precision = precision_score(y_test, predictions_with_outliers, average='weighted')
cnn_recall = recall_score(y_test, predictions_with_outliers, average='weighted')
cnn_f1 = f1_score(y_test, predictions_with_outliers, average='weighted')


cnn_outliers = np.where(predictions_with_outliers == 10)[0]
ocsvm_outliers = np.where(y_test == 10)[0]
lof_outliers = np.where(lof_labels == 0)[0]
iforest_outliers = np.where(iforest_labels == 0)[0]
envelope_outliers = np.where(envelope_labels == 0)[0]
cnn_only_outliers = np.setdiff1d(cnn_outliers, np.union1d(np.union1d(np.union1d(ocsvm_outliers, lof_outliers), iforest_outliers), envelope_outliers))

# Terminal
print(f"OCSVM Total Detected Outliers: {ocsvm_total_outliers}")
for i in range(10):
    print(f"OCSVM Detected Outliers in Class {i}: {ocsvm_per_class_outliers[i]}")
print(f"LOF Detected Outliers: {np.sum(lof_labels == 0)}")
print(f"Isolation Forest Detected Outliers: {np.sum(iforest_labels == 0)}")
print(f"Elliptic Envelope Detected Outliers: {np.sum(envelope_labels == 0)}")
print(f"CNN Detected Outliers: {len(cnn_outliers)}")
print(f"CNN Only Detected Outliers: {len(cnn_only_outliers)}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {cnn_precision:.4f}")
print(f"Recall: {cnn_recall:.4f}")
print(f"F1-Score: {cnn_f1:.4f}")

# Graph
fig, ax = plt.subplots(figsize=(12, 10))

outlier_indices = np.where(y_with_outliers == 10)[0]
normal_indices = np.where(y_with_outliers != 10)[0]

# Points
ax.scatter(X_pca[normal_indices, 0], X_pca[normal_indices, 1], c='black', s=10, label='Normal')
ax.scatter(X_pca[ocsvm_outliers, 0], X_pca[ocsvm_outliers, 1], c='red', s=10, label='OCSVM Outliers')
ax.scatter(X_pca[lof_outliers, 0], X_pca[lof_outliers, 1], c='red', s=10, label='LOF Outliers')
ax.scatter(X_pca[iforest_outliers, 0], X_pca[iforest_outliers, 1], c='red', s=10, label='Isolation Forest Outliers')
ax.scatter(X_pca[envelope_outliers, 0], X_pca[envelope_outliers, 1], c='red', s=10, label='Elliptic Envelope Outliers')
ax.scatter(X_pca[cnn_only_outliers, 0], X_pca[cnn_only_outliers, 1], c='blue', s=10, label='CNN Extra Outliers')

ax.set_title("Outlier Detection Results")
ax.legend()

total_outliers = len(np.union1d(np.union1d(np.union1d(ocsvm_outliers, lof_outliers), iforest_outliers), np.union1d(envelope_outliers, cnn_only_outliers)))

print(f"Total Detected Outliers by All Methods: {total_outliers}")

text = (f"OCSVM Detected Outliers: {ocsvm_total_outliers}\n"
         f"LOF Detected Outliers: {np.sum(lof_labels == 0)}\n"
         f"Isolation Forest Detected Outliers: {np.sum(iforest_labels == 0)}\n"
         f"Elliptic Envelope Detected Outliers: {np.sum(envelope_labels == 0)}\n"
         f"CNN Detected Outliers: {len(cnn_outliers)}\n"
         f"Total Detected Outliers by All Methods: {total_outliers}\n"
         f"Model Accuracy: {test_accuracy:.4f}")
ax.text(0.5, -0.2, text, transform=ax.transAxes, ha='center', va='top', fontsize=12)

plt.tight_layout()
plt.show()