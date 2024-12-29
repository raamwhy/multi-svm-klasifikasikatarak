import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set randomness globally
np.random.seed(42)

# Load dataset
file_path = "Klasifikasi Katarak.xlsx"  
data = pd.read_excel(file_path)

# Cek jumlah data yang dibaca
print(f"Jumlah data yang dibaca dari Excel: {data.shape[0]}")  

# Praproses data
data.columns = ['Feature1', 'Feature2', 'Feature3', 'Label']  
label_encoder = LabelEncoder()
data['EncodedLabel'] = label_encoder.fit_transform(data['Label'])

X = data[['Feature1', 'Feature2', 'Feature3']]
y = data['EncodedLabel']

# Standardisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduksi Dimensi menggunakan PCA (jika perlu)
pca = PCA(n_components=3, random_state=42) 
X_pca = pca.fit_transform(X_scaled)


# Split data untuk training dan testing
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_pca, y, np.arange(len(y)), test_size=0.1, random_state=42
)

# Menampilkan jumlah data
print(f"Total data: {len(y)}") 
print(f"Jumlah data training: {len(X_train)}")  
print(f"Jumlah data testing: {len(X_test)}")   

# Model SVM dengan hyperparameter terbaik
svm_model = SVC(C=1000, gamma=1, kernel='rbf', probability=True, decision_function_shape='ovr', random_state=42)

# Evaluasi awal pada data training dan testing
y_train_pred_initial = svm_model.fit(X_train, y_train).predict(X_train)
y_test_pred_initial = svm_model.predict(X_test)

train_accuracy_initial = accuracy_score(y_train, y_train_pred_initial)
test_accuracy_initial = accuracy_score(y_test, y_test_pred_initial)

print(f"\nAkurasi data training sebelum K-fold: {train_accuracy_initial * 100:.2f}%")
print(f"Akurasi data testing sebelum K-fold: {test_accuracy_initial * 100:.2f}%")

# Cross-validation dengan 10 fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\nHasil Validasi Silang (10 Fold):")
cv_scores = []
best_accuracy = 0
best_model = None
best_fold = None
best_train_indices = None
best_test_indices = None

for fold, (train_idx, test_idx) in enumerate(cv.split(X_pca, y), start=1):
    X_train_fold, X_test_fold = X_pca[train_idx], X_pca[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    # Melatih model pada fold ini
    svm_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = svm_model.predict(X_test_fold)
    accuracy = accuracy_score(y_test_fold, y_pred_fold)
    cv_scores.append(accuracy)

    # Laporan klasifikasi dan matriks kebingungan
    class_report = classification_report(y_test_fold, y_pred_fold, target_names=label_encoder.classes_)
    conf_matrix_fold = confusion_matrix(y_test_fold, y_pred_fold)

    # Menampilkan hasil fold
    print(f"Fold {fold}:")
    print(f"  Indeks data training: {train_idx}")
    print(f"  Indeks data testing: {test_idx}")
    print(f"  Akurasi: {accuracy * 100:.2f}%")
    print(f"\nLaporan Klasifikasi (Fold {fold}):")
    print(class_report)
    print(f"\nMatriks Kebingungan (Fold {fold}):")
    print(conf_matrix_fold)

    # Simpan model dan akurasi terbaik
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = svm_model  
        best_fold = fold  
        best_train_indices = train_idx  
        best_test_indices = test_idx 

# Rata-rata akurasi dari K-fold
mean_cv_accuracy = np.mean(cv_scores)
print(f"\nRata-rata akurasi dari K-fold: {mean_cv_accuracy * 100:.2f}%")
print(f"Akurasi tertinggi dari K-fold (Fold {best_fold}): {best_accuracy * 100:.2f}%")

# Menampilkan data Fold terbaik
print("\nIndeks data yang digunakan untuk training pada Fold dengan akurasi terbaik:")
print(best_train_indices)

print("\nIndeks data yang digunakan untuk testing pada Fold dengan akurasi terbaik:")
print(best_test_indices)

# Evaluasi model pada data testing menggunakan model terbaik
y_test_pred_final = best_model.predict(X_test)
test_accuracy_final = accuracy_score(y_test, y_test_pred_final)

# Menghitung akurasi pada data training untuk model terbaik
y_train_pred_final = best_model.predict(X_train)
train_accuracy_final = accuracy_score(y_train, y_train_pred_final)

print(f"\nAkurasi data training (model terbaik): {train_accuracy_final * 100:.2f}%")
print(f"Akurasi data testing (model terbaik): {test_accuracy_final * 100:.2f}%")

# Laporan klasifikasi
print("\nLaporan Klasifikasi (Testing):")
print(classification_report(y_test, y_test_pred_final, target_names=label_encoder.classes_))

# Matriks kebingungan untuk data testing
conf_matrix_test = confusion_matrix(y_test, y_test_pred_final)
print("\nMatriks Kebingungan (Testing):")
print(conf_matrix_test)


# Menghitung sensitivitas dan spesifisitas untuk setiap kelas
num_classes = conf_matrix_test.shape[0]
sensitivity = []
specificity = []

for i in range(num_classes):
    TN = conf_matrix_test.sum() - conf_matrix_test[i, :].sum() - conf_matrix_test[:, i].sum() + conf_matrix_test[i, i]
    FP = conf_matrix_test[:, i].sum() - conf_matrix_test[i, i]
    FN = conf_matrix_test[i, :].sum() - conf_matrix_test[i, i]
    TP = conf_matrix_test[i, i]

    sensitivity.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
    specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)

# Menampilkan sensitivitas dan spesifisitas untuk setiap kelas
print("\nSensitivitas dan Spesifisitas untuk setiap kelas:")
for i in range(num_classes):
    print(f"Kelas {label_encoder.classes_[i]}:")
    print(f"  Sensitivitas (Recall): {sensitivity[i] * 100:.2f}%")
    print(f"  Spesifisitas: {specificity[i] * 100:.2f}%")

def predict_user_input():
    try:
        print("\nMasukkan nilai fitur citra untuk prediksi:")
        feature1 = float(input("Feature1: "))  # Input dari pengguna untuk Feature1
        feature2 = float(input("Feature2: "))  # Input dari pengguna untuk Feature2
        feature3 = float(input("Feature3: "))  # Input dari pengguna untuk Feature3
    except ValueError:
        print("Input tidak valid. Pastikan memasukkan angka.")
        return

    # Standarisasi input pengguna
    user_input = np.array([[feature1, feature2, feature3]])  
    user_input_scaled = scaler.transform(user_input)  

    # Reduksi dimensi input menggunakan PCA
    user_input_pca = pca.transform(user_input_scaled)

    # Prediksi dengan model SVM dari Fold terbaik
    prediction = best_model.predict(user_input_pca)

    # Tampilkan hasil prediksi
    predicted_class = label_encoder.inverse_transform(prediction)
    print(f"\nHasil prediksi untuk citra dengan fitur yang dimasukkan: {predicted_class[0]}")

# Menjalankan fungsi prediksi berdasarkan input pengguna
predict_user_input()

# Plot grafik akurasi tiap fold
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='b', label='Akurasi per Fold')
plt.axhline(y=mean_cv_accuracy, color='r', linestyle='--', label=f'Rata-rata Akurasi: {mean_cv_accuracy * 100:.2f}%')
plt.title('Akurasi Per Fold pada Cross-Validation (10 Fold)')
plt.xlabel('Fold')
plt.ylabel('Akurasi (%)')
plt.xticks(range(1, len(cv_scores) + 1))
plt.legend()
plt.grid(True)
plt.show()
