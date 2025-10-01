import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Veri yükle (örnek)
df = pd.read_csv('water_potability.csv')

# Özellik ve hedef ayır
X = df.drop('Potability', axis=1)
y = df['Potability']

# Eksik değer doldurma, ön işleme vs. (sadece örnek)
X.fillna(X.mean(), inplace=True)

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modeller
knn = KNeighborsClassifier()
svm = SVC(probability=True)
rf = RandomForestClassifier()

# Model eğitimi
knn.fit(X_scaled, y)
svm.fit(X_scaled, y)
rf.fit(X_scaled, y)

# ANN modeli oluşturma ve eğitim
ann = Sequential()
ann.add(Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)))
ann.add(Dense(16, activation='relu'))
ann.add(Dense(1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_scaled, y, epochs=10, batch_size=16)

# Model ve scaler kayıt
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(knn, 'knn_model.joblib')
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(rf, 'rf_model.pkl')
ann.save('ann_model.h5')
