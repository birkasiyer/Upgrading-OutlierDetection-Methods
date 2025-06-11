import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LogisticRegression

# Veriyi yükleyip PCA ile 2D'ye indirgeme
data = load_breast_cancer()
X = data.data
y = data.target

# Veri Normalizasyonu veya Standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Özellik Seçimi - PCA (Boyut indirgeme)
pca = PCA(n_components=2)  # PCA ile veriyi 2D'ye indirgeme
X_pca = pca.fit_transform(X_scaled)

# Aykırı Değer Tespitine başlamadan önce veriyi 2D'ye indirgemek için PCA kullanalım
X_final = X_pca

# OCSVM (One-Class SVM)
ocsvm = OneClassSVM(nu=0.05)
ocsvm_outliers = ocsvm.fit_predict(X_final)

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
dbscan = DBSCAN(eps=1.5, min_samples=8)
dbscan_outliers = dbscan.fit_predict(X_final)

# Elliptic Envelope
elliptic = EllipticEnvelope(contamination=0.05)
elliptic_outliers = elliptic.fit_predict(X_final)

# Meta model için özellik matrisi oluşturma
meta_features = np.vstack((ocsvm_outliers == -1, dbscan_outliers == -1, elliptic_outliers == -1)).T

# Logistic Regression modelini eğitmek için y'nin tersini kullanıyoruz
meta_target = (y == 0).astype(int)  # Aykırı değerleri 1, normal değerleri 0 olarak ayarlıyoruz

# Logistic Regression meta modelini eğitme
meta_model = LogisticRegression()
meta_model.fit(meta_features, meta_target)

# Meta model ile tahmin yapma
meta_outliers = meta_model.predict(meta_features)

# Ortak aykırı değerler ve birleşik aykırı değerler Logistic Regression meta modeli dahil edilerek hesaplanır
combined_outliers = (ocsvm_outliers == -1) | (dbscan_outliers == -1) | (elliptic_outliers == -1) | (meta_outliers == 1)
common_outliers = (ocsvm_outliers == -1) & (dbscan_outliers == -1) & (elliptic_outliers == -1) & (meta_outliers == 1)

# Precision, Recall, F1-Score hesaplama ve yazdırma
def print_metrics(model_name, y_true, y_pred, ax):
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    ax.text(0.5, -0.25, f"F1-Score: {f1*100:.2f}%",
            ha='center', transform=ax.transAxes, fontsize=8)
    print(f"{model_name} : Precision: {precision*100:.2f}% , Recall: {recall*100:.2f}% , F1-Score: {f1*100:.2f}%")


# Grafik Oluşturma
fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # Grafik boyutunu küçültme
axs = axs.flatten()

# OCSVM Grafiği
axs[0].scatter(X_final[:, 0], X_final[:, 1], color='b', label='Normal', s=8)
axs[0].scatter(X_final[ocsvm_outliers == -1, 0], X_final[ocsvm_outliers == -1, 1], color='r', label='Outliers', s=8)
axs[0].set_title('One-Class SVM (OCSVM)', fontsize=10)  # Başlık fontunu küçültme
axs[0].legend(fontsize=8)  # Legend fontunu küçültme
axs[0].text(0.5, -0.2, f"Outliers: {np.sum(ocsvm_outliers == -1)}",
               transform=axs[0].transAxes, ha='center', fontsize=8)
print_metrics("One-Class SVM (OCSVM)", y, ocsvm_outliers, axs[0])

# DBSCAN Grafiği
axs[1].scatter(X_final[:, 0], X_final[:, 1], color='b', label='Normal', s=8)
axs[1].scatter(X_final[dbscan_outliers == -1, 0], X_final[dbscan_outliers == -1, 1], color='r', label='Outliers', s=8)
axs[1].set_title('DBSCAN (Density-Based)', fontsize=10)
axs[1].legend(fontsize=8)
axs[1].text(0.5, -0.2, f"Outliers: {np.sum(dbscan_outliers== -1)}",
               transform=axs[1].transAxes, ha='center', fontsize=8)
print_metrics("DBSCAN (Density-Based)", y, dbscan_outliers, axs[1])

# Elliptic Envelope Grafiği
axs[2].scatter(X_final[:, 0], X_final[:, 1], color='b', label='Normal', s=8)
axs[2].scatter(X_final[elliptic_outliers == -1, 0], X_final[elliptic_outliers == -1, 1], color='r', label='Outliers', s=8)
axs[2].set_title('Elliptic Envelope', fontsize=10)
axs[2].legend(fontsize=8)
axs[2].text(0.5, -0.2, f"Outliers: {np.sum(elliptic_outliers == -1)}",
               transform=axs[2].transAxes, ha='center', fontsize=8)
print_metrics("Elliptic Envelope", y, elliptic_outliers, axs[2])

# Logistic Regression tabanlı meta model grafiği
axs[3].scatter(X_final[:, 0], X_final[:, 1], color='b', label='Normal', s=8)
axs[3].scatter(X_final[meta_outliers == 1, 0], X_final[meta_outliers == 1, 1], color='r', label='Outliers', s=8)
axs[3].set_title('Meta Model (Logistic Regression)', fontsize=10)
axs[3].legend(fontsize=8)
combined_outliers_count = np.sum(combined_outliers)
common_outliers_count = np.sum(common_outliers)
axs[3].text(0.5, -0.25,
            f"Meta Model Outliers: {np.sum(meta_outliers == 1)}\n All Combined Outliers: {combined_outliers_count}\n All Common Outliers: {common_outliers_count}\n",
            ha='center', transform=axs[3].transAxes, fontsize=8)
print_metrics("Meta Model (Logistic Regression)", meta_target, meta_outliers, axs[3])

plt.tight_layout()
plt.show()

# Aykırı değerlerin sayısını yazdırma
print("OCSVM - Outliers:", np.sum(ocsvm_outliers == -1))
print("DBSCAN - Outliers:", np.sum(dbscan_outliers == -1))
print("Elliptic Envelope - Outliers:", np.sum(elliptic_outliers == -1))
print("Common Outliers (All Methods):", np.sum(common_outliers))
print("Combined Outliers (All Methods):", np.sum(combined_outliers))