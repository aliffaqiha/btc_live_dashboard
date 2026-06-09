import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import zipfile
import kaggle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Bitcoin Analytics Dashboard", layout="wide")
st.title("Bitcoin Historical Data & AI Forecasting Dashboard")
st.markdown("Analisis pergerakan harga Bitcoin & Prediksi AI (Multivariate LSTM). By Alif Faqih")

# --- 1. FUNGSI LOAD DATA (CACHED) ---
@st.cache_data(ttl=86400)
def load_data_from_kaggle(filename):
    dataset_slug = "novandraanugrah/bitcoin-historical-datasets-2018-2024"
    if not os.path.exists(filename):
        with st.spinner(f"Mengunduh data {filename} dari Kaggle..."):
            try:
                kaggle.api.dataset_download_file(dataset_slug, filename, path='.')
                if os.path.exists(filename + ".zip"):
                    with zipfile.ZipFile(filename + ".zip", 'r') as zip_ref:
                        zip_ref.extractall('.')
                    os.remove(filename + ".zip")
            except Exception as e:
                st.error(f"Gagal: {e}")
                return None
    df = pd.read_csv(filename)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df.set_index('Open time', inplace=True)
    df.sort_index(inplace=True)
    return df

# --- 2. FUNGSI MODEL MULTIVARIATE LSTM (REVISED) ---
@st.cache_resource
def train_and_predict_lstm(df, look_back=60):
    # Salin dataframe agar tidak merubah data asli di dashboard
    df_model = df.copy()
    
    # 1. Transformasi stasioner: Log Return untuk Harga, Pct Change untuk Volume
    df_model['Log_Ret'] = np.log(df_model['Close'] / df_model['Close'].shift(1))
    df_model['Vol_Change'] = df_model['Volume'].pct_change()
    df_model.dropna(inplace=True)
    
    # Ambil 2 fitur utama: Log Return Harga dan Perubahan Volume
    feature_data = df_model[['Log_Ret', 'Vol_Change']].values.astype('float32')
    
    # Normalisasi fitur ke skala (-1, 1) agar seimbang
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(feature_data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])       # 2 Fitur masuk ke time window
        y.append(scaled_data[i, 0])                 # Target hanya memprediksi Log_Ret (Kolom 0)
    X, y = np.array(X), np.array(y)
    
    # Arsitektur Deep LSTM dengan Dropout penangkal garis lurus
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(look_back, 2)),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    
    # Proses Forecasting Rekursif 30 Langkah ke depan
    last_window = scaled_data[-look_back:]
    predicted_log_returns = []
    
    for _ in range(30):
        # Prediksi Log Return satu langkah ke depan
        pred = model.predict(last_window.reshape(1, look_back, 2), verbose=0)
        pred_value = pred[0, 0]
        predicted_log_returns.append(pred_value)
        
        # Untuk langkah berikutnya, asumsikan perubahan volume stabil (0)
        next_step = np.array([[pred_value, 0.0]], dtype='float32')
        last_window = np.append(last_window[1:], next_step, axis=0)
        
    # Denormalisasi hasil prediksi log return kembali ke skala aslinya
    dummy_array = np.zeros((len(predicted_log_returns), 2))
    dummy_array[:, 0] = predicted_log_returns
    denormalized = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Rekonstruksi dari Nilai Log Return menjadi Nilai Dolar ($) Asli Bitcoin
    last_actual_price = df['Close'].iloc[-1]
    predicted_prices = []
    current_price = last_actual_price
    
    for log_ret in denormalized:
        current_price = current_price * np.exp(log_ret)
        predicted_prices.append(current_price)
        
    return np.array(predicted_prices).reshape(-1, 1)

# --- 3. UI SIDEBAR ---
st.sidebar.header("Pengaturan")
timeframe = st.sidebar.selectbox("Pilih Timeframe:", ["Harian (1D)", "4 Jam (4H)", "1 Jam (1H)"])
file_mapping = {
    "Harian (1D)": "btc_1d_data_2018_to_2025.csv",
    "4 Jam (4H)": "btc_4h_data_2018_to_2025.csv",
    "1 Jam (1H)": "btc_1h_data_2018_to_2025.csv"
}

df_btc = load_data_from_kaggle(file_mapping[timeframe])

if df_btc is not None:
    # Filter Tahun
    years = st.sidebar.slider("Rentang Tahun:", int(df_btc.index.year.min()), int(df_btc.index.year.max()), (2022, 2025))
    df_f = df_btc[(df_btc.index.year >= years[0]) & (df_btc.index.year <= years[1])]

    # Metrik
    c1, c2, c3 = st.columns(3)
    c1.metric("Harga Close Terakhir", f"${df_f['Close'].iloc[-1]:,.2f}")
    c2.metric("Tertinggi", f"${df_f['High'].max():,.2f}")
    c3.metric("Terendah", f"${df_f['Low'].min():,.2f}")

    # Grafik Candlestick
    st.subheader("Grafik Harga")
    fig = go.Figure(data=[go.Candlestick(x=df_f.index, open=df_f['Open'], high=df_f['High'], low=df_f['Low'], close=df_f['Close'])])
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, width='stretch')

    # --- 4. INTEGRASI LSTM ---
    st.markdown("---")
    st.subheader("Prediksi Harga (LSTM)")
    if st.button("Jalankan Forecast"):
        with st.spinner("Model sedang mempelajari korelasi pergerakan harga dan volume..."):
            pred = train_and_predict_lstm(df_f)
            forecast_dates = pd.date_range(start=df_f.index[-1], periods=31)[1:]
            
            fig_ai = go.Figure()
            fig_ai.add_trace(go.Scatter(x=df_f.tail(60).index, y=df_f['Close'].tail(60), name="Data Asli"))
            fig_ai.add_trace(go.Scatter(x=forecast_dates, y=pred.flatten(), name="Prediksi AI", line=dict(color='red', dash='dash')))
            fig_ai.update_layout(template="plotly_dark")
            st.plotly_chart(fig_ai, width='stretch')
            st.success("Prediksi selesai dengan pemrosesan Multivariate Log-Return!")
else:
    st.error("Data tidak ditemukan.")