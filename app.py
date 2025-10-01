import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from scipy import stats

# Veri Yükle ve Ön İşle
df = pd.read_csv("water_potability.csv")

# Eksik değer doldurma
imputer = SimpleImputer(strategy='mean')
df[df.columns[:-1]] = imputer.fit_transform(df[df.columns[:-1]])

# Aykırı değer temizleme (Z-score)
z_scores = np.abs(stats.zscore(df.iloc[:, :-1]))
df = df[(z_scores < 3).all(axis=1)]

# Giriş ve hedef ayrımı
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE ile dengeleme
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Eğitim
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# Streamlit Arayüzü
st.title("💧 Dünya Su Kalitesi Tahmini")

st.markdown("Lütfen aşağıdaki verileri girerek suyun içilebilir olup olmadığını tahmin edin:")

ph = st.slider("pH", min_value=0.0, max_value=14.0, value=7.0)
Hardness = st.slider("Hardness", 0.0, 400.0, 150.0)
Solids = st.slider("Solids (ppm)", 0.0, 50000.0, 20000.0)
Chloramines = st.slider("Chloramines", 0.0, 15.0, 5.0)
Sulfate = st.slider("Sulfate", 0.0, 500.0, 200.0)
Conductivity = st.slider("Conductivity", 0.0, 1000.0, 400.0)
Organic_carbon = st.slider("Organic Carbon", 0.0, 30.0, 10.0)
Trihalomethanes = st.slider("Trihalomethanes", 0.0, 125.0, 60.0)
Turbidity = st.slider("Turbidity", 0.0, 10.0, 3.0)

user_input = np.array([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity,
                        Organic_carbon, Trihalomethanes, Turbidity]])

# Veriyi ölçekle
user_input_scaled = scaler.transform(user_input)

# Tahmin
if st.button("Tahmin Et"):
    prediction = model.predict(user_input_scaled)
    proba = model.predict_proba(user_input_scaled)  # Olasılık tahmini
    confidence = proba[0][1] * 100  # İçilebilir sınıfının olasılığı yüzde olarak

    if prediction[0] == 1:
        st.success(f"💧 Bu su içilebilir. (Güven: %{confidence:.2f})")
    else:
        st.error(f"🚱 Bu su içilemez. (İçilebilirlik olasılığı: %{confidence:.2f})")
