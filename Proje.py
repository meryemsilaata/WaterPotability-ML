

# GEREKLİ KÜTÜPHANELER
import pandas as pd
import numpy as np#sayısal 
import matplotlib.pyplot as plt#Grafik çizimi ve görselleştirme için kullanılır.
import seaborn as sns#korelasyon
from sklearn.model_selection import train_test_split, cross_val_score#Veriyi eğitim ve test olarak bölme, çapraz doğrulama gibi işlemleri sağlar
from sklearn.preprocessing import StandardScaler#Verinin ölçeklendirilmesi ve ön işleme işlemleri için.
from sklearn.impute import SimpleImputer#	Eksik değerlerin doldurulması için yöntemler içerir
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
#Model performans ölçümleri (accuracy, precision, recall vb.) ve ROC eğrisi hesaplamaları için kullanılır.

from sklearn.neighbors import KNeighborsClassifier#	K-En Yakın Komşu (KNN) sınıflandırıcı
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE#	SMOTE algoritması ile dengesiz veri setlerinde veri artırımı yapılır
from scipy import stats#	İstatistiksel işlemler, aykırı değer analizi için kullanılır.
#Yapay Sinir Ağı (ANN) oluşturmak ve eğitmek için derin öğrenme kütüphanesi.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping



# VERİYİ YÜKLE

df = pd.read_csv("water_potability.csv")

# VERİ ARTTIRMA
artiracak_miktar = 4000
rastgele_yeni_ornekler = df.sample(n=artiracak_miktar, replace=True, random_state=42)#sample() ile veri setinden rastgele, tekrarlı 2000 yeni örnek alınır
df = pd.concat([df, rastgele_yeni_ornekler], ignore_index=True)
print(f"Veri artırma sonrası toplam örnek sayısı: {df.shape[0]}")

print("Mevcut sütunlar:", df.columns)


# ÖN İŞLEME ÖNCESİ KORELASYON MATRİSİ
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Ön İşleme Öncesi Korelasyon Matrisi")
plt.show()

# EKSİK VERİ ANALİZİ ve SIFIR DEĞERLERİ NaN YAPMA
sifir_olmamasi_gereken = ['ph', 'Sulfate', 'Chloramines', 'Trihalomethanes']
for col in sifir_olmamasi_gereken:
    sifir_sayisi = (df[col] == 0).sum()
    print(f"{col} sütunundaki sıfır sayısı (NaN yapılacak): {sifir_sayisi}")
    df[col] = df[col].apply(lambda x: np.nan if x == 0 else x)

# EKSİK DEĞERLERİ MEDYAN İLE DOLDUR
eksik_degerler = df.isna().sum()#(aykırı değerlere karşı sağlam yöntem)
print("\nMedyan ile doldurmadan önce eksik değer sayıları:\n", eksik_degerler[sifir_olmamasi_gereken])

imputer = SimpleImputer(strategy='median')
df[df.columns[:-1]] = imputer.fit_transform(df[df.columns[:-1]])

eksik_degerler_son = df.isna().sum()
print("\nMedyan ile doldurduktan sonra eksik değer sayıları:\n", eksik_degerler_son[sifir_olmamasi_gereken])



# Z-SCORE İLE AYKIRI DEĞER TEMİZLİĞİ
z_scores = np.abs(stats.zscore(df.iloc[:, :-1]))#Böylece modelin aşırı uç verilere duyarlılığı azaltılır
aykiri_satir_sayisi = df.shape[0] - df[(z_scores < 3).all(axis=1)].shape[0]
print(f"\nZ-Score ile temizlenecek aykırı değer içeren satır sayısı: {aykiri_satir_sayisi}")

df = df[(z_scores < 3).all(axis=1)]
print(f"Aykırı değer temizliği sonrası veri seti boyutu: {df.shape}")


# NORMALİZASYON  Veri ölçekleme: StandardScaler ile normalize edildi.
X = df.drop("Potability", axis=1)#StandardScaler ile tüm özellikleri aynı ölçeğe getirerek algoritmanın tüm özelliklere eşit ağırlık vermesini sağlarız
y = df["Potability"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE ÖNCESİ SINIF DAĞILIMI
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette='Set2')
plt.title('SMOTE Öncesi Sınıf Dağılımı')
plt.xlabel('Potability (0 = İçilemez, 1 = İçilebilir)')
plt.ylabel('Örnek Sayısı')
plt.grid(axis='y')
plt.show()

# SMOTE İLE VERİ DENGELEME
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# SMOTE SONRASI SINIF DAĞILIMI
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled, palette='Set3')
plt.title('SMOTE Sonrası Sınıf Dağılımı')
plt.xlabel('Potability (0 = İçilemez, 1 = İçilebilir)')
plt.ylabel('Örnek Sayısı')
plt.grid(axis='y')
plt.show()

# KORELASYON MATRİSİ
df_smote = pd.DataFrame(X_resampled, columns=X.columns)
df_smote["Potability"] = y_resampled
plt.figure(figsize=(10, 8))
sns.heatmap(df_smote.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Ön İşleme Sonrası Korelasyon Matrisi")
plt.show()

# GÖRSELLEŞTİRME - Histogram
df_smote.drop("Potability", axis=1).hist(figsize=(12, 10), bins=30)
plt.suptitle("Özelliklerin Dağılımı - Histogram")
plt.tight_layout()
plt.show()

# 5. Boxplotlar ile Aykırı Değer Görselleştirme
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()  # sayısal sütunları seçer

# Eğer hedef sütun numeric_columns içindeyse, çıkar
if "Potability" in numeric_columns:
    numeric_columns.remove("Potability")


plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(f"{col} - Boxplot")
    plt.tight_layout()#çakısma önler düzen sağlar
plt.show()

# VERİYİ EĞİTİM VE TESTE AYIR
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# MODELLER
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),#eğitim verisindeki en yakın 5 komşunun sınıfına göre sınıflandırılır.
    "SVM": SVC(kernel='rbf', C=1, gamma='scale'),# C=1 hiperparametresi overfitting ve margin genişliği arasında denge sağlar.
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)#
}

# MODEL EĞİTİMİ VE PERFORMANS
results = []#Her modelin performans metriklerini saklamak için boş liste.
all_preds = {}#Her modelin test verisine yaptığı tahminleri saklar.

for name, model in models.items():
    model.fit(X_train, y_train)#Model eğitim verisi üzerinde öğrenme yapar.
    preds = model.predict(X_test)#Model, test verisi üzerinde tahmin yapar
    all_preds[name] = preds
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),#Genel doğru tahmin oranı
        "Precision": precision_score(y_test, preds),#Pozitif tahminlerin doğruluğu
        "Recall": recall_score(y_test, preds),#Gerçek pozitifleri yakalama oranı
        "F1 Score": f1_score(y_test, preds)#Precision ve Recall dengesi
    })

# RANDOM FOREST İÇİN CROSS-VALIDATION (: 5 katlı çapraz doğrulama yapar. Veri 5 parçaya bölünür, her seferinde biri test verisi olur)
rf = RandomForestClassifier(n_estimators=100, random_state=42)#Ormanda 100 adet karar ağacı 
cv_scores = cross_val_score(rf, X_resampled, y_resampled, cv=5)# SMOTE sonrası dengelenmiş veri seti.
print("Random Forest - 5-Fold Cross Validation Accuracy Ortalaması: %.4f" % np.mean(cv_scores))#Tüm katların doğruluk ortalaması hesaplanır.

# GELİŞTİRİLMİŞ ANN MODELİ
ann = Sequential([#Katmanları sıralı şekilde tanımlayan bir yapıdır
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),#Öğrenmeyi hızlandırır, kararlı hale getirir.
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),#Overfitting’i önlemek için kullanılır.
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = ann.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)#verbos çıktı göstermez(sessiz mod)

# ANN Tahmin
ann_probs = ann.predict(X_test)
ann_preds = (ann_probs > 0.5).astype(int)
all_preds["ANN"] = ann_preds.ravel()

results.append({
    "Model": "ANN",
    "Accuracy": accuracy_score(y_test, ann_preds),
    "Precision": precision_score(y_test, ann_preds),
    "Recall": recall_score(y_test, ann_preds),
    "F1 Score": f1_score(y_test, ann_preds)
})

# SONUÇLAR
results_df = pd.DataFrame(results)
print("\nModel Performans Karşılaştırması:\n")
print(results_df.round(4).sort_values("F1 Score", ascending=False))

# MODELLERİN F1 SKOR GRAFİĞİ
sns.set(style="whitegrid")
plt.figure(figsize=(10, 4))
sns.barplot(data=results_df.sort_values("F1 Score", ascending=False), x="Model", y="F1 Score", palette="viridis")
plt.title("Modellerin F1 Skor Karşılaştırması")
plt.ylabel("F1 Skoru")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.show()

# KARMAŞIKLIK MATRİSLERİ
for name, preds in all_preds.items():
    cm = confusion_matrix(y_test, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Karmaşıklık Matrisi")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.show()
    


# ANN - Epoch-Loss Eğrisi
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı', linewidth=2)
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı', linewidth=2)
plt.title('Yapay Sinir Ağı - Epoch Loss Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.legend()
plt.grid(True)
plt.show()


# ROC EĞRİSİ - ANN
fpr, tpr, _ = roc_curve(y_test, ann_probs)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, ann_probs):.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrisi - ANN")
plt.legend()
plt.grid(True)
plt.show()

# ROC Eğrisi - KNN, SVM, Random Forest için
from sklearn.metrics import roc_auc_score, roc_curve

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]  # Olasılık tahminlerinin pozitif sınıf için olanı
    else:
        # SVM için predict_proba yoksa decision_function kullan
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # Normalize
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Eğrisi - {name}")
    plt.legend()
    plt.grid(True)
    plt.show()










# Tüm modeller için ROC AUC hesapla
roc_results = []

for name, preds in all_preds.items():
    if name == "ANN":
        probs = ann_probs.ravel()
    else:
        if hasattr(models[name], "predict_proba"):
            probs = models[name].predict_proba(X_test)[:, 1]
        else:  # SVC gibi modellerde predict_proba olmayabilir
            try:
                probs = models[name].decision_function(X_test)
                probs = (probs - probs.min()) / (probs.max() - probs.min())  # normalize et
            except:
                probs = preds  # fallback: tahmin kullan (doğru değil ama hata vermez)

    auc_score = roc_auc_score(y_test, probs)
    roc_results.append({"Model": name, "ROC AUC": auc_score})

# ROC AUC skorlarını ekle
roc_df = pd.DataFrame(roc_results)
results_df = results_df.merge(roc_df, on="Model")

print("\nROC AUC Skorları:\n")
print(results_df[["Model", "ROC AUC"]].sort_values("ROC AUC", ascending=False).round(4))

# Tüm modeller için ROC eğrilerini çiz
plt.figure(figsize=(10, 6))
for name, preds in all_preds.items():
    if name == "ANN":
        probs = ann_probs.ravel()
    else:
        if hasattr(models[name], "predict_proba"):
            probs = models[name].predict_proba(X_test)[:, 1]
        else:
            try:
                probs = models[name].decision_function(X_test)
                probs = (probs - probs.min()) / (probs.max() - probs.min())
            except:
                continue  # ROC eğrisi çizilemeyenleri atla

    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Tüm Modeller - ROC Eğrileri")
plt.legend()
plt.grid(True)
plt.show()




