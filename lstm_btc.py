"""
Bitcoin Historical Analytics Dashboard
Fitur: Auto-fetch Kaggle Data via API (Set-and-Forget)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import zipfile

# --- 1. KONFIGURASI KAGGLE API ---
try:
    if hasattr(st, "secrets") and "KAGGLE_USERNAME" in st.secrets:
        os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
        os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
except Exception:
    # Jika di lokal dan st.secrets eror/tidak ada, abaikan saja
    # Karena Kaggle API akan otomatis mencari file kaggle.json di folder lokal
    pass

import kaggle

# Konfigurasi halaman utama dashboard
st.set_page_config(page_title="Bitcoin Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("Bitcoin Historical Data Interactive Dashboard")
st.markdown("Dashboard analisis pergerakan harga Bitcoin secara real-time berdasarkan pembaruan data otomatis dari Kaggle.\n By Alif Faqih")

# --- 2. FUNGSI SMART DOWNLOAD DATA (CACHED 24 JAM) ---
@st.cache_data(ttl=86400) # Data disimpan di memori selama 24 jam / otomatis fetch ulang data terbaru
def load_data_from_kaggle(filename):
    dataset_slug = "novandraanugrah/bitcoin-historical-datasets-2018-2024"

    if not os.path.exists(filename):
        with st.spinner(f"🔄 Mengambil data baru `{filename}` langsung dari Kaggle API..."):
            try:
                # Download file langsung dari Kaggle
                kaggle.api.dataset_download_file(dataset_slug, filename, path='.')

                if os.path.exists(filename + ".zip"):
                    with zipfile.ZipFile(filename + ".zip", 'r') as zip_ref:
                        zip_ref.extractall('.')
                    os.remove(filename + ".zip") # Hapus zip
            except Exception as e:
                st.error(f"Gagal mengunduh data: {e}")
                return None

    # Membaca file CSV ke dalam Pandas Dataframe
    df = pd.read_csv(filename)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df.set_index('Open time', inplace=True)
    df.sort_index(inplace=True)
    return df

# --- 3. SIDEBAR KONTROL USER ---
st.sidebar.header("Pengaturan Dashboard")

timeframe = st.sidebar.selectbox(
    "Pilih Timeframe Data:",
    ["15 Menit (15m)","Harian (1D)", "4 Jam (4H)", "1 Jam (1H)"]
)

# Pemetaan nama file asli di Kaggle
file_mapping = {
    "Harian (1D)": "btc_1d_data_2018_to_2025.csv",
    "4 Jam (4H)": "btc_4h_data_2018_to_2025.csv",
    "1 Jam (1H)": "btc_1h_data_2018_to_2025.csv",
    "15 Menit (15m)": "btc_15m_data_2018_to_2025.csv"
}

target_file = file_mapping[timeframe]
df_btc = load_data_from_kaggle(target_file)

if df_btc is not None:
    # Filter Tahun Berdasarkan Slider
    min_year = int(df_btc.index.year.min())
    max_year = int(df_btc.index.year.max())
    
    selected_years = st.sidebar.slider(
        "Pilih Rentang Tahun:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Filter data berdasarkan tahun terpilih
    df_filtered = df_btc[(df_btc.index.year >= selected_years[0]) & (df_btc.index.year <= selected_years[1])]

    # --- 4. TAMPILAN METRIK UTAMA (KPI) ---
    latest_close = df_filtered['Close'].iloc[-1]
    highest_price = df_filtered['High'].max()
    lowest_price = df_filtered['Low'].min()
    total_volume = df_filtered['Volume'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Harga Close Terakhir", f"${latest_close:,.2f}")
    col2.metric("Harga Tertinggi (Periode)", f"${highest_price:,.2f}")
    col3.metric("Harga Terendah (Periode)", f"${lowest_price:,.2f}")
    col4.metric("Total Volume Transaksi", f"{total_volume:,.0f} BTC")

    st.markdown("---")

    # --- 5. GRAFIK INTERAKTIF CANDLESTICK ---
    st.subheader("Grafik Pergerakan Harga Candlestick")
    
    # Batasi lilin agar performa browser tetap cepat dan tidak lag
    df_candle = df_filtered.tail(1000) if len(df_filtered) > 1000 else df_filtered

    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_candle.index,
        open=df_candle['Open'],
        high=df_candle['High'],
        low=df_candle['Low'],
        close=df_candle['Close'],
        name="BTC"
    )])
    fig_candle.update_layout(xaxis_rangeslider_visible=True, template="plotly_dark", height=500)
    st.plotly_chart(fig_candle, use_container_width=True)

    # --- 6. ANALISIS VOLUME & DISTRIBUSI ---
    st.markdown("---")
    st.subheader("Analisis Volume & Aktivitas Pasar")
    
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Tren Volume Dagang")
        fig_vol = px.line(df_filtered, x=df_filtered.index, y='Volume', template="plotly_dark", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_right:
        st.markdown("#### Taker Buy vs Total Volume (Aktivitas Buyer Agresif)")
        df_filtered['Taker_Ratio'] = df_filtered['Taker buy base asset volume'] / (df_filtered['Volume'] + 1e-8)
        fig_taker = px.histogram(
            df_filtered, 
            x='Taker_Ratio', 
            nbins=50, 
            color_discrete_sequence=['#FF9900'],
            template="plotly_dark"
        )
        st.plotly_chart(fig_taker, use_container_width=True)

    # --- 7. TABEL DATA MENTAH ---
    st.markdown("---")
    st.subheader("100 Data Transaksi Terakhir")
    st.dataframe(df_filtered.tail(100), use_container_width=True)
else:
    st.error("Gagal memuat data dari Kaggle. Pastikan token Kaggle API Anda valid.")