import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Input
from sklearn.metrics import precision_score, recall_score, f1_score
import keras_tuner as kt

digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

ocsvm_per_class_outliers = {}
ocsvm_total_outliers = 0
ocsvm_labels_all = np.zeros_like(y)

y_ocsvm_eval = np.ones_like(y)

param_grid = {'nu': [0.01, 0.05, 0.7, 0.1], 'kernel': ['poly', 'rbf', 'sigmoid', 'linear']}

def ocswm_score_func(estimator, X):
    return np.mean(estimator.predict(X) == 1)

ocsvm_grid = GridSearchCV(OneClassSVM(), param_grid, cv=3, scoring=ocswm_score_func)
ocsvm_grid.fit(X_scaled)

best_ocsvm = ocsvm_grid.best_estimator_

for i in range(10):
    class_indices = np.where(y == i)[0]
    X_class = X_scaled[class_indices]

    ocsvm_outliers = best_ocsvm.fit_predict(X_class)

    ocsvm_labels = np.where(ocsvm_outliers == 1, 1, 0)
    ocsvm_labels_all[class_indices] = ocsvm_labels

    y_ocsvm_eval[class_indices[ocsvm_labels == 0]] = 0

    ocsvm_per_class_outliers[i] = np.sum(ocsvm_labels == 0)
    ocsvm_total_outliers += np.sum(ocsvm_labels == 0)

ocsvm_precision = precision_score(y_ocsvm_eval, ocsvm_labels_all, pos_label=0)
ocsvm_recall = recall_score(y_ocsvm_eval, ocsvm_labels_all, pos_label=0)
ocsvm_f1 = f1_score(y_ocsvm_eval, ocsvm_labels_all, pos_label=0)

y_with_outliers = np.where(ocsvm_labels_all == 0, 10, y)  # -1 for outliers -> 10 for outliers

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_with_outliers, test_size=0.3, random_state=42)

class CNNHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))

        model.add(Dense(units=hp.Int('units_1', min_value=64, max_value=256, step=64), activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

        model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu'))
        model.add(Dense(11, activation='softmax'))  # 10 classes (0-9) + 1 for outliers (10)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

tuner = kt.RandomSearch(
    CNNHyperModel(),
    objective=kt.Objective("val_accuracy", direction="max"),
    max_trials=20,
    executions_per_trial=3,
    directory='my_dir',
    project_name='digit_classifier'
)

tuner.search(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

best_cnn_model = tuner.get_best_models(num_models=1)[0]

test_loss, test_accuracy = best_cnn_model.evaluate(X_test, y_test, verbose=1)

predictions = np.argmax(best_cnn_model.predict(X_test), axis=1)

predictions_with_outliers = np.where(predictions == 10, 10, predictions)

cnn_precision = precision_score(y_test, predictions_with_outliers, average='weighted')
cnn_recall = recall_score(y_test, predictions_with_outliers, average='weighted')
cnn_f1 = f1_score(y_test, predictions_with_outliers, average='weighted')

cnn_outliers = np.where(predictions_with_outliers == 10)[0]
ocsvm_outliers = np.where(y_test == 10)[0]
cnn_only_outliers = np.setdiff1d(cnn_outliers, ocsvm_outliers)

print("=== OCSVM Outlier Detection Results ===")
for i in range(10):
    print(f"Class {i} - OCSVM detected {ocsvm_per_class_outliers[i]} outliers")

print(f"\n=== Total OCSVM Detected Outliers ===")
print(f"Total OCSVM Outliers: {ocsvm_total_outliers}")
print(f"OCSVM Precision: {ocsvm_precision:.2f}")
print(f"OCSVM Recall: {ocsvm_recall:.2f}")
print(f"OCSVM F1-Score: {ocsvm_f1:.2f}")

print(f"\n=== CNN Test Results ===")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {cnn_precision:.2f}")
print(f"Recall: {cnn_recall:.2f}")
print(f"F1-Score: {cnn_f1:.2f}")

print(f"\n=== CNN Detected Outliers that OCSVM did not detect ===")
print(f"CNN detected {len(cnn_only_outliers)} additional outliers not detected by OCSVM")

fig, ax = plt.subplots(2, 1, figsize=(10, 12))

outlier_indices = np.where(y_with_outliers == 10)[0]
normal_indices = np.where(y_with_outliers != 10)[0]

ax[0].scatter(X_pca[normal_indices, 0], X_pca[normal_indices, 1], c='blue', s=10, label='Normal')
ax[0].scatter(X_pca[outlier_indices, 0], X_pca[outlier_indices, 1], c='red', s=10, label='Outliers')

ax[0].set_title("OCSVM Results (PCA Reduced Data)", fontsize=14)
ax[0].set_xlabel("PCA1")
ax[0].set_ylabel("PCA2")
ax[0].legend(loc='upper right')
ax[0].grid(True)

text_str = (
    f"=== Total OCSVM Results ===\n"
    f"Outliers Detected: {len(outlier_indices)}\n"
    f"Normal Data Points: {len(X_scaled) - len(outlier_indices)}\n\n"
    f"OCSVM Precision: {ocsvm_precision:.2f}\n"
    f"OCSVM Recall: {ocsvm_recall:.2f}\n"
    f"OCSVM F1-Score: {ocsvm_f1:.2f}\n\n"
    f"=== CNN Test Accuracy ===\n"
    f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    f"Precision: {cnn_precision:.2f}\n"
    f"Recall: {cnn_recall:.2f}\n"
    f"F1-Score: {cnn_f1:.2f}\n"
    f"Total OCSVM Outliers: {ocsvm_total_outliers}\n"
    f"CNN Detected Outliers not caught by OCSVM: {len(cnn_only_outliers)}"
)
ax[1].text(0.5, 0.8, text_str, wrap=True, horizontalalignment='center', fontsize=12, verticalalignment='bottom')
ax[1].axis('off')

plt.tight_layout()
plt.show()