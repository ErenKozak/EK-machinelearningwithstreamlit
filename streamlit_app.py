import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Örnek veri seti oluşturma
np.random.seed(42)
data = {
    "square_meters": np.random.randint(50, 300, 100),
    "num_rooms": np.random.randint(1, 6, 100),
    "age": np.random.randint(0, 50, 100),
    "price": np.random.randint(100000, 1000000, 100)
}
df = pd.DataFrame(data)

# Modeli eğitme
X = df[["square_meters", "num_rooms", "age"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit arayüzü
st.title("Ev Fiyat Tahmini Uygulaması")
st.write("Lütfen evin özelliklerini girin ve fiyat tahminini görün.")

square_meters = st.number_input("Evin metrekaresi", min_value=20, max_value=500, value=100)
num_rooms = st.number_input("Oda sayısı", min_value=1, max_value=10, value=3)
age = st.number_input("Evin yaşı", min_value=0, max_value=100, value=10)

if st.button("Fiyat Tahmini Yap"):
    input_data = np.array([[square_meters, num_rooms, age]])
    prediction = model.predict(input_data)[0]
    st.success(f"Tahmini Ev Fiyatı: {prediction:,.0f} TL")

# Model performansını gösterme
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Modelin Ortalama Mutlak Hata (MAE) Değeri: {mae:,.0f} TLdir")
