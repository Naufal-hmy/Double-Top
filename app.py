import streamlit as st
import pandas as pd
import numpy as np
import scipy.signal
import plotly.graph_objects as go
import os

# Set page layout to wide and dark theme by default
st.set_page_config(
    page_title="Dashboard Bursa Cerdas - Deteksi Double Top",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS injection (Bursa style)
st.markdown("""
    <style>
    .main {
        background-color: #111111;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #1a1a1a;
        color: #ffffff;
        border-right: 1px solid #222222;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00ff88 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .stMetric {
        background-color: #222222;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #333333;
    }
    .stMetric label {
        color: #888888 !important;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    .stDataFrame {
        border: 1px solid #333333;
        border-radius: 6px;
    }
    button[data-baseweb="tab"] {
        color: #888888 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00ff88 !important;
        border-bottom-color: #00ff88 !important;
    }
    
    /* Mobile-responsiveness overrides */
    @media (max-width: 768px) {
        h1 {
            font-size: 24px !important;
            text-align: center !important;
        }
        h2 {
            font-size: 20px !important;
        }
        h3 {
            font-size: 16px !important;
        }
        .stMetric {
            padding: 10px !important;
            margin-bottom: 10px !important;
        }
        .stMetric div[data-testid="stMetricValue"] {
            font-size: 18px !important;
        }
        .js-plotly-plot {
            max-width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)


# Path defaults
csv_path = "transaksi_harian_202606130928.csv"
path_lokal = r"c:\Users\41123\Documents\Semester 6\Sistem Cerdas\transaksi_harian_202606130928.csv"

# ==============================================================================
# DATA INGESTION & PREPROCESSING
# ==============================================================================
# Commented out cache to force reload and clean data
def load_and_clean_data():
    target_path = path_lokal if os.path.exists(path_lokal) else csv_path
    if not os.path.exists(target_path):
        return None
        
    df = pd.read_csv(target_path, sep=';')
    # Clean headers
    df.columns = df.columns.str.replace('"', '').str.strip()
    # Clean text values
    df['kode'] = df['kode'].astype(str).str.replace('"', '').str.strip().str.upper()
    df['tanggal'] = df['tanggal'].astype(str).str.replace('"', '').str.strip()
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    
    # Cast to numeric
    for col in ['open_price', 'close_price', 'high_price', 'low_price', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Clean invalid/zero prices
    # 1. Fill missing/zero close_price (cannot have a trade without a close price)
    df = df.dropna(subset=['close_price'])
    df = df[df['close_price'] > 0]
    
    # 2. If open_price is 0 or NaN, set it to close_price
    df['open_price'] = df['open_price'].fillna(df['close_price'])
    df.loc[df['open_price'] <= 0, 'open_price'] = df['close_price']
    
    # 3. If high_price is 0, NaN, or lower than close/open, bound it
    df['high_price'] = df['high_price'].fillna(df['close_price'])
    df.loc[df['high_price'] <= 0, 'high_price'] = df['close_price']
    df['high_price'] = df[['high_price', 'open_price', 'close_price']].max(axis=1)
    
    # 4. If low_price is 0, NaN, or higher than close/open, bound it
    df['low_price'] = df['low_price'].fillna(df['close_price'])
    df.loc[df['low_price'] <= 0, 'low_price'] = df['close_price']
    df['low_price'] = df[['low_price', 'open_price', 'close_price']].min(axis=1)
    
    df = df.drop_duplicates(subset=['kode', 'tanggal'])
    df = df.sort_values(by=['kode', 'tanggal']).reset_index(drop=True)
    return df

df = load_and_clean_data()

if df is None:
    st.error(f"Berkas '{csv_path}' tidak ditemukan di folder Anda. Pastikan file CSV diletakkan di direktori yang sama dengan script ini.")
    st.stop()

# ==============================================================================
# ALGORITMA DETEKSI DOUBLE TOP
# ==============================================================================

# Algoritma A: Scipy Peak Prominence
def detect_scipy(df_ticker, distance, tolerance, prior_days, prior_pct):
    df_ticker = df_ticker.sort_values('tanggal').reset_index(drop=True)
    close = df_ticker['close_price'].values
    high = df_ticker['high_price'].values
    low = df_ticker['low_price'].values
    dates = df_ticker['tanggal'].tolist()
    
    prominence = 0.03 * np.mean(close)
    peaks, _ = scipy.signal.find_peaks(high, distance=distance, prominence=prominence)
    
    double_tops = []
    for i in range(len(peaks) - 1):
        p1_idx = peaks[i]
        p2_idx = peaks[i+1]
        
        p1_price = high[p1_idx]
        p2_price = high[p2_idx]
        
        # Height tolerance
        if abs(p1_price - p2_price) / max(p1_price, p2_price) > tolerance:
            continue
            
        # Trough search
        trough_sub = low[p1_idx:p2_idx]
        trough_price = min(trough_sub)
        trough_idx = p1_idx + np.argmin(trough_sub)
        
        # Depth check
        depth = (min(p1_price, p2_price) - trough_price) / min(p1_price, p2_price)
        if depth < 0.03 or depth > 0.35:
            continue
            
        # Prior trend
        start_idx = max(0, p1_idx - prior_days)
        if start_idx == p1_idx:
            continue
        min_prior = np.min(low[start_idx:p1_idx])
        if min_prior <= 0 or (p1_price - min_prior) / min_prior < prior_pct:
            continue
            
        # Breakout
        breakout_idx = None
        for j in range(p2_idx + 1, min(len(df_ticker), p2_idx + 40)):
            if high[j] > max(p1_price, p2_price) * 1.05:
                break
            if close[j] < trough_price:
                breakout_idx = j
                break
                
        if breakout_idx is None:
            continue
            
        # Backtest outcomes
        pattern_height = (p1_price + p2_price) / 2 - trough_price
        target_price = trough_price - pattern_height
        stop_loss = trough_price + 0.5 * pattern_height
        
        success = None
        outcome_idx = None
        for k in range(breakout_idx + 1, min(len(df_ticker), breakout_idx + 60)):
            if low[k] <= target_price:
                success = True
                outcome_idx = k
                break
            if high[k] >= stop_loss:
                success = False
                outcome_idx = k
                break
                
        double_tops.append({
            'ticker': df_ticker['kode'].iloc[0],
            'p1_idx': int(p1_idx), 'p1_date': dates[p1_idx], 'p1_price': float(p1_price),
            'p2_idx': int(p2_idx), 'p2_date': dates[p2_idx], 'p2_price': float(p2_price),
            'trough_idx': int(trough_idx), 'trough_date': dates[trough_idx], 'trough_price': float(trough_price),
            'breakout_idx': int(breakout_idx), 'breakout_date': dates[breakout_idx], 'breakout_price': float(close[breakout_idx]),
            'success': success, 'outcome_idx': outcome_idx, 'outcome_date': dates[outcome_idx] if outcome_idx is not None else None,
            'target_price': float(target_price), 'stop_loss_price': float(stop_loss)
        })
    return double_tops

# Algoritma B: Rolling Window Extremum
def detect_rolling(df_ticker, distance, tolerance, prior_days, prior_pct):
    df_ticker = df_ticker.sort_values('tanggal').reset_index(drop=True)
    close = df_ticker['close_price'].values
    high = df_ticker['high_price'].values
    low = df_ticker['low_price'].values
    dates = df_ticker['tanggal'].tolist()
    n = len(df_ticker)
    
    # Locate local peaks using rolling window
    peaks = []
    for i in range(distance, n - distance):
        window_highs = high[i - distance : i + distance + 1]
        if high[i] == max(window_highs):
            # Check if it stands out from local average
            avg_local = np.mean(close[i - distance : i + distance + 1])
            if high[i] > avg_local * 1.03:
                peaks.append(i)
                
    double_tops = []
    for i in range(len(peaks) - 1):
        p1_idx = peaks[i]
        p2_idx = peaks[i+1]
        
        p1_price = high[p1_idx]
        p2_price = high[p2_idx]
        
        if abs(p1_price - p2_price) / max(p1_price, p2_price) > tolerance:
            continue
            
        trough_sub = low[p1_idx:p2_idx]
        trough_price = min(trough_sub)
        trough_idx = p1_idx + np.argmin(trough_sub)
        
        depth = (min(p1_price, p2_price) - trough_price) / min(p1_price, p2_price)
        if depth < 0.03 or depth > 0.35:
            continue
            
        start_idx = max(0, p1_idx - prior_days)
        if start_idx == p1_idx:
            continue
        min_prior = np.min(low[start_idx:p1_idx])
        if min_prior <= 0 or (p1_price - min_prior) / min_prior < prior_pct:
            continue
            
        breakout_idx = None
        for j in range(p2_idx + 1, min(n, p2_idx + 40)):
            if high[j] > max(p1_price, p2_price) * 1.05:
                break
            if close[j] < trough_price:
                breakout_idx = j
                break
                
        if breakout_idx is None:
            continue
            
        pattern_height = (p1_price + p2_price) / 2 - trough_price
        target_price = trough_price - pattern_height
        stop_loss = trough_price + 0.5 * pattern_height
        
        success = None
        outcome_idx = None
        for k in range(breakout_idx + 1, min(n, breakout_idx + 60)):
            if low[k] <= target_price:
                success = True
                outcome_idx = k
                break
            if high[k] >= stop_loss:
                success = False
                outcome_idx = k
                break
                
        double_tops.append({
            'ticker': df_ticker['kode'].iloc[0],
            'p1_idx': int(p1_idx), 'p1_date': dates[p1_idx], 'p1_price': float(p1_price),
            'p2_idx': int(p2_idx), 'p2_date': dates[p2_idx], 'p2_price': float(p2_price),
            'trough_idx': int(trough_idx), 'trough_date': dates[trough_idx], 'trough_price': float(trough_price),
            'breakout_idx': int(breakout_idx), 'breakout_date': dates[breakout_idx], 'breakout_price': float(close[breakout_idx]),
            'success': success, 'outcome_idx': outcome_idx, 'outcome_date': dates[outcome_idx] if outcome_idx is not None else None,
            'target_price': float(target_price), 'stop_loss_price': float(stop_loss)
        })
    return double_tops

# Algoritma C: ZigZag Indicator Pattern Matching
def detect_zigzag(df_ticker, distance, tolerance, prior_days, prior_pct, change_pct=0.05):
    df_ticker = df_ticker.sort_values('tanggal').reset_index(drop=True)
    close = df_ticker['close_price'].values
    high = df_ticker['high_price'].values
    low = df_ticker['low_price'].values
    dates = df_ticker['tanggal'].tolist()
    n = len(df_ticker)
    
    zigzag_points = []
    last_val = close[0]
    last_idx = 0
    is_up = True
    
    for i in range(1, n):
        curr_val = close[i]
        diff = (curr_val - last_val) / last_val
        
        if is_up:
            if curr_val > high[last_idx]:
                last_idx = i
                last_val = curr_val
            elif diff < -change_pct:
                zigzag_points.append(('peak', last_idx, high[last_idx]))
                is_up = False
                last_idx = i
                last_val = curr_val
        else:
            if curr_val < low[last_idx]:
                last_idx = i
                last_val = curr_val
            elif diff > change_pct:
                zigzag_points.append(('trough', last_idx, low[last_idx]))
                is_up = True
                last_idx = i
                last_val = curr_val
                
    double_tops = []
    for idx in range(len(zigzag_points) - 2):
        p1 = zigzag_points[idx]
        tr = zigzag_points[idx+1]
        p2 = zigzag_points[idx+2]
        
        if p1[0] == 'peak' and tr[0] == 'trough' and p2[0] == 'peak':
            p1_idx, p1_price = p1[1], p1[2]
            p2_idx, p2_price = p2[1], p2[2]
            trough_idx, trough_price = tr[1], tr[2]
            
            if abs(p1_price - p2_price) / max(p1_price, p2_price) > tolerance:
                continue
                
            if (p2_idx - p1_idx) < distance:
                continue
                
            depth = (min(p1_price, p2_price) - trough_price) / min(p1_price, p2_price)
            if depth < 0.03 or depth > 0.35:
                continue
                
            start_idx = max(0, p1_idx - prior_days)
            if start_idx == p1_idx:
                continue
            min_prior = np.min(low[start_idx:p1_idx])
            if min_prior <= 0 or (p1_price - min_prior) / min_prior < prior_pct:
                continue
                
            breakout_idx = None
            for j in range(p2_idx + 1, min(n, p2_idx + 40)):
                if high[j] > max(p1_price, p2_price) * 1.05:
                    break
                if close[j] < trough_price:
                    breakout_idx = j
                    break
                    
            if breakout_idx is None:
                continue
                
            pattern_height = (p1_price + p2_price) / 2 - trough_price
            target_price = trough_price - pattern_height
            stop_loss = trough_price + 0.5 * pattern_height
            
            success = None
            outcome_idx = None
            for k in range(breakout_idx + 1, min(n, breakout_idx + 60)):
                if low[k] <= target_price:
                    success = True
                    outcome_idx = k
                    break
                if high[k] >= stop_loss:
                    success = False
                    outcome_idx = k
                    break
                    
            double_tops.append({
                'ticker': df_ticker['kode'].iloc[0],
                'p1_idx': int(p1_idx), 'p1_date': dates[p1_idx], 'p1_price': float(p1_price),
                'p2_idx': int(p2_idx), 'p2_date': dates[p2_idx], 'p2_price': float(p2_price),
                'trough_idx': int(trough_idx), 'trough_date': dates[trough_idx], 'trough_price': float(trough_price),
                'breakout_idx': int(breakout_idx), 'breakout_date': dates[breakout_idx], 'breakout_price': float(close[breakout_idx]),
                'success': success, 'outcome_idx': outcome_idx, 'outcome_date': dates[outcome_idx] if outcome_idx is not None else None,
                'target_price': float(target_price), 'stop_loss_price': float(stop_loss)
            })
    return double_tops

# ==============================================================================
# SIDEBAR CONTROLS & SCANNING
# ==============================================================================
st.sidebar.markdown("<h2 style='text-align: center; color: #00ff88;'>PARAMETER MODEL</h2>", unsafe_allow_html=True)

# Hardcoded model parameters for simplified UTS presentation
dist_val = 15
tol_val = 0.04
trend_pct = 0.10
prior_days_val = 25

# ==============================================================================
# PRE-CALCULATE OVERALL STATS
# ==============================================================================
@st.cache_data
def scan_all_stocks_combined(dist, tol, prior_days, prior_pct):
    all_patterns = []
    tickers = df['kode'].unique()
    
    for ticker in tickers:
        df_ticker = df[df['kode'] == ticker]
        if len(df_ticker) >= 60:
            # Detect using Scipy
            pats_scipy = detect_scipy(df_ticker, dist, tol, prior_days, prior_pct)
            for p in pats_scipy:
                p['algo'] = "Scipy Peak Prominence"
            all_patterns.extend(pats_scipy)
            
            # Detect using Rolling Window
            pats_rolling = detect_rolling(df_ticker, dist, tol, prior_days, prior_pct)
            for p in pats_rolling:
                p['algo'] = "Rolling Window Extremum"
            all_patterns.extend(pats_rolling)
            
            # Detect using ZigZag
            pats_zigzag = detect_zigzag(df_ticker, dist, tol, prior_days, prior_pct)
            for p in pats_zigzag:
                p['algo'] = "ZigZag Pattern Matching"
            all_patterns.extend(pats_zigzag)
            
    return all_patterns

all_detected = scan_all_stocks_combined(dist_val, tol_val, prior_days_val, trend_pct)

# Get Tickers
demo_tickers = sorted(list(set([p['ticker'] for p in all_detected])))

# Helper to calculate stats per ticker
def get_ticker_stats(ticker, patterns):
    t_pats = [p for p in patterns if p['ticker'] == ticker]
    total = len(t_pats)
    resolved = [p for p in t_pats if p['success'] is not None]
    sukses = sum(1 for p in resolved if p['success'] == True)
    gagal = sum(1 for p in resolved if p['success'] == False)
    aktif = total - len(resolved)
    akurasi = (sukses / len(resolved) * 100) if len(resolved) > 0 else 0.0
    return {
        'total': total,
        'aktif': aktif,
        'sukses': sukses,
        'gagal': gagal,
        'akurasi': akurasi
    }

# Pre-calculate stats for sorting and selectbox label
ticker_stats = {}
for t in demo_tickers:
    ticker_stats[t] = get_ticker_stats(t, all_detected)

# Sort tickers: highest accuracy first, then most TP, then alphabetical
demo_tickers_sorted = sorted(
    demo_tickers,
    key=lambda x: (ticker_stats[x]['akurasi'], ticker_stats[x]['sukses'], x),
    reverse=True
)

# Filter Emiten berdasarkan Status Pola
filter_status = st.sidebar.radio(
    "Filter Emiten Berdasarkan Status Pola:",
    ["Semua Emiten", "Hanya Sinyal Aktif (Ongoing)"]
)

# Terapkan penyaringan pada daftar emiten
if filter_status == "Hanya Sinyal Aktif (Ongoing)":
    active_tickers = list(set([p['ticker'] for p in all_detected if p['success'] is None]))
    demo_tickers_display = [t for t in demo_tickers_sorted if t in active_tickers]
else:
    demo_tickers_display = demo_tickers_sorted

# Let user choose emiten
if demo_tickers_display:
    selected_emiten = st.sidebar.selectbox(
        "Pilih Emiten untuk Grafik:",
        demo_tickers_display,
        format_func=lambda x: f"{x} (Akurasi: {ticker_stats[x]['akurasi']:.0f}% | {ticker_stats[x]['sukses']} TP / {ticker_stats[x]['gagal']} SL)"
    )
else:
    st.sidebar.info("Tidak ada emiten dengan Sinyal Aktif (Ongoing) saat ini.")
    selected_emiten = "TIDAK ADA SAHAM TERDETEKSI"



# ==============================================================================
# MAIN DASHBOARD PAGE
# ==============================================================================
st.markdown("<h1 style='text-align: center; color: #00ff88; margin-bottom: 20px;'>DASHBOARD BURSA CERDAS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888; font-size: 14px;'>Sistem Kecerdasan Buatan Deteksi & Backtesting Pola Bearish Reversal \"Double Top\" Saham Indonesia (IDX)</p>", unsafe_allow_html=True)

# 1. KINERJA GLOBAL ALGORITMA
st.markdown("<h4 style='color: #00ff88; margin-bottom: 10px;'>Kinerja Global Algoritma (Seluruh Saham IDX)</h4>", unsafe_allow_html=True)
g_col1, g_col2, g_col3, g_col4 = st.columns(4)

global_total = len(all_detected)
global_resolved = [p for p in all_detected if p['success'] is not None]
global_success = sum(1 for p in global_resolved if p['success'] == True)
global_fail = sum(1 for p in global_resolved if p['success'] == False)
global_active = global_total - len(global_resolved)
global_accuracy = (global_success / len(global_resolved) * 100) if len(global_resolved) > 0 else 0.0

with g_col1:
    st.metric("Total Pola (Global)", f"{global_total} Pola")
with g_col2:
    st.metric("Sinyal Aktif (Global)", f"{global_active} Pola")
with g_col3:
    st.metric("Akurasi Sinyal (Global)", f"{global_accuracy:.2f}%")
with g_col4:
    st.metric("Perbandingan TP vs SL (Global)", f"{global_success} : {global_fail}")

st.markdown("<br>", unsafe_allow_html=True)

# 2. KINERJA EMITEN PILIHAN
st.markdown(f"<h4 style='color: #00ff88; margin-bottom: 10px;'>Kinerja Khusus Saham Pilihan ({selected_emiten})</h4>", unsafe_allow_html=True)
e_col1, e_col2, e_col3, e_col4 = st.columns(4)

if selected_emiten != "TIDAK ADA SAHAM TERDETEKSI" and demo_tickers:
    stats = ticker_stats[selected_emiten]
    total_detected = stats['total']
    active_count = stats['aktif']
    success_count = stats['sukses']
    fail_count = stats['gagal']
    accuracy = stats['akurasi']
else:
    total_detected = 0
    active_count = 0
    success_count = 0
    fail_count = 0
    accuracy = 0.0

with e_col1:
    st.metric("Total Pola Saham", f"{total_detected} Pola")
with e_col2:
    st.metric("Sinyal Aktif Saham", f"{active_count} Pola")
with e_col3:
    st.metric("Akurasi Sinyal Saham", f"{accuracy:.2f}%")
with e_col4:
    st.metric("Perbandingan TP vs SL Saham", f"{success_count} : {fail_count}")

# Helper to render the candlestick chart and details for a specific algorithm
def render_algorithm_chart(algo_name, pattern, df_ticker, key_suffix):
    if pattern is not None:
        status_pola = "SUKSES" if pattern['success'] == True else ("GAGAL" if pattern['success'] == False else "BELUM SELESAI")
        color = "#ffd700" if pattern['success'] == True else ("#ff3333" if pattern['success'] == False else "#888888")
        st.markdown(f"### {algo_name} - <span style='color:{color}; font-weight:bold;'>{status_pola}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"### {algo_name}")
        st.info(f"Tidak ada pola Double Top terdeteksi oleh algoritma **{algo_name}** pada tanggal breakout ini.")
        return
        
    p1_idx = pattern['p1_idx']
    p2_idx = pattern['p2_idx']
    trough_idx = pattern['trough_idx']
    breakout_idx = pattern['breakout_idx']
    outcome_idx = pattern['outcome_idx']
    
    start_plot = max(0, p1_idx - 30)
    end_plot = min(len(df_ticker), (outcome_idx if outcome_idx is not None else breakout_idx) + 30)
    df_sub = df_ticker.iloc[start_plot:end_plot].reset_index(drop=True)
    
    # Convert dates to string format to eliminate weekend/holiday gaps
    df_sub['tanggal_str'] = df_sub['tanggal'].dt.strftime('%Y-%m-%d')
    
    p1_p = p1_idx - start_plot
    p2_p = p2_idx - start_plot
    tr_p = trough_idx - start_plot
    br_p = breakout_idx - start_plot
    out_p = (outcome_idx - start_plot) if outcome_idx is not None else None
    
    p1_date_str = df_sub['tanggal_str'].iloc[p1_p]
    p2_date_str = df_sub['tanggal_str'].iloc[p2_p]
    tr_date_str = df_sub['tanggal_str'].iloc[tr_p]
    br_date_str = df_sub['tanggal_str'].iloc[br_p]
    out_date_str = df_sub['tanggal_str'].iloc[out_p] if out_p is not None else None
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df_sub['tanggal_str'],
        open=df_sub['open_price'],
        high=df_sub['high_price'],
        low=df_sub['low_price'],
        close=df_sub['close_price'],
        name=f'{pattern["ticker"]} Candlestick',
        increasing=dict(line=dict(color='#00ff88')),
        decreasing=dict(line=dict(color='#ff3333'))
    ))
    
    fig.add_trace(go.Scatter(
        x=[p1_date_str, p2_date_str],
        y=[pattern['p1_price'], pattern['p2_price']],
        mode='markers+text',
        marker=dict(color='#ff3333', size=13, symbol='triangle-down'),
        text=['Puncak 1', 'Puncak 2'],
        textposition='top center',
        name='Resistance Peak',
        textfont=dict(color='#ffffff')
    ))
    
    fig.add_trace(go.Scatter(
        x=[tr_date_str],
        y=[pattern['trough_price']],
        mode='markers+text',
        marker=dict(color='#3399ff', size=11, symbol='triangle-up'),
        text=['Lembah (Neckline)'],
        textposition='bottom center',
        name='Neckline Support',
        textfont=dict(color='#ffffff')
    ))
    
    fig.add_trace(go.Scatter(
        x=[p1_date_str, br_date_str],
        y=[pattern['trough_price'], pattern['trough_price']],
        mode='lines',
        line=dict(color='#3399ff', width=2, dash='dash'),
        name='Garis Neckline (Batas Support)'
    ))
    
    fig.add_trace(go.Scatter(
        x=[br_date_str],
        y=[pattern['breakout_price']],
        mode='markers+text',
        marker=dict(color='orange', size=12, symbol='x'),
        text=['BREAKOUT! (Sinyal Jual)'],
        textposition='bottom right',
        name='Sinyal Breakout',
        textfont=dict(color='orange')
    ))
    
    x_dates_post = df_sub['tanggal_str'].iloc[br_p:]
    fig.add_trace(go.Scatter(
        x=x_dates_post,
        y=[pattern['target_price']] * len(x_dates_post),
        mode='lines',
        line=dict(color='#ffd700', width=2, dash='dot'),
        name='Target Turun (Take Profit)'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_dates_post,
        y=[pattern['stop_loss_price']] * len(x_dates_post),
        mode='lines',
        line=dict(color='#ff3333', width=2, dash='dot'),
        name='Stop Loss (SL)'
    ))
    
    if out_date_str is not None:
        status_txt = "TP TERCAPAI (SUKSES)" if pattern['success'] else "SL TERKENA (GAGAL)"
        warna = '#ffd700' if pattern['success'] else '#ff3333'
        fig.add_trace(go.Scatter(
            x=[out_date_str],
            y=[df_sub['close_price'].iloc[out_p]],
            mode='markers+text',
            marker=dict(color=warna, size=15, symbol='star' if pattern['success'] else 'octagon'),
            text=[status_txt],
            textposition='top right',
            name='Outcome',
            textfont=dict(color=warna)
        ))
        
    fig.add_vrect(
        x0=p1_date_str,
        x1=br_date_str,
        fillcolor='rgba(255, 0, 0, 0.05)',
        layer='below',
        line_width=0,
        annotation_text='Fase Pembentukan (M-Shape)',
        annotation_position='top left',
        annotation_font=dict(color='#ff3333', size=11)
    )
    
    end_outcome_str = out_date_str if out_date_str is not None else df_sub['tanggal_str'].iloc[-1]
    fig.add_vrect(
        x0=br_date_str,
        x1=end_outcome_str,
        fillcolor='rgba(255, 215, 0, 0.04)' if pattern['success'] else 'rgba(128, 128, 128, 0.04)',
        layer='below',
        line_width=0,
        annotation_text='Fase Proyeksi Turun',
        annotation_position='top left',
        annotation_font=dict(color='#ffd700' if pattern['success'] else '#888888', size=11)
    )
    
    fig.add_vline(
        x=br_date_str,
        line_width=2,
        line_dash='dash',
        line_color='orange',
        annotation_text='<b>Pola Resmi Terjadi (Sinyal Jual)</b>',
        annotation_position='top',
        annotation_font=dict(color='orange', size=11)
    )
    
    status_pola_info = "SUKSES" if pattern['success'] == True else ("GAGAL" if pattern['success'] == False else "BELUM SELESAI")
    
    fig.update_layout(
        yaxis_title="Harga Saham (IDR)",
        xaxis_title="Tanggal Perdagangan",
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        margin=dict(t=20, b=30, l=40, r=40),
        showlegend=False,
        height=450
    )
    
    # Force X-axis to be a category axis to remove weekend/holiday gaps
    fig.update_xaxes(
        type='category',
        tickangle=-45,
        nticks=10
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{key_suffix}")
    
    p1_dt_str = pd.to_datetime(pattern['p1_date']).strftime('%d %B %Y')
    br_dt_str = pd.to_datetime(pattern['breakout_date']).strftime('%d %B %Y')
    pct_drop = abs(pattern['target_price'] - pattern['breakout_price']) / pattern['breakout_price'] * 100
    
    st.info(f"""
        **Keterangan ({algo_name})**:
        - Pola mulai terdeteksi sejak **{p1_dt_str}** (Puncak 1 terbentuk di harga Rp {pattern['p1_price']:,}).
        - Pola **resmi terkonfirmasi** pada tanggal **{br_dt_str}** ketika harga saham ditutup di bawah garis support leher (Rp {pattern['trough_price']:,}) di harga Rp {pattern['breakout_price']:,}. Ini adalah momen keluarnya **Sinyal Jual**.
        - Harga target penurunan diprediksi jatuh ke area **Rp {pattern['target_price']:,}** (Potensi penurunan harga sebesar **{pct_drop:.2f}%** dari breakout).
        - Status Backtest: **{status_pola_info}**
    """)

# Create Tabs
tab1, tab2 = st.tabs(["📊 Visualisasi Grafik", "📋 Hasil Prediksi & Data"])

# TAB 1: VISUALISASI GRAFIK
with tab1:
    if selected_emiten == "TIDAK ADA SAHAM TERDETEKSI" or not demo_tickers:
        st.warning("Tidak ada emiten yang mendeteksi pola Double Top dengan kombinasi parameter saat ini. Silakan melonggarkan parameter model.")
    else:
        df_ticker = df[df['kode'] == selected_emiten].reset_index(drop=True)
        ticker_patterns = [p for p in all_detected if p['ticker'] == selected_emiten]
        
        # Split patterns by algorithm
        pats_scipy = [p for p in ticker_patterns if p['algo'] == "Scipy Peak Prominence"]
        pats_rolling = [p for p in ticker_patterns if p['algo'] == "Rolling Window Extremum"]
        pats_zigzag = [p for p in ticker_patterns if p['algo'] == "ZigZag Pattern Matching"]
        
        # Calculate stats for labeling
        stats_scipy = get_ticker_stats(selected_emiten, pats_scipy)
        stats_rolling = get_ticker_stats(selected_emiten, pats_rolling)
        stats_zigzag = get_ticker_stats(selected_emiten, pats_zigzag)
        
        # Get unique breakout dates for this stock across all algorithms
        unique_breakouts = sorted(list(set([pd.to_datetime(p['breakout_date']).strftime('%Y-%m-%d') for p in ticker_patterns])), reverse=True)
        
        # Build descriptive labels for dropdown
        breakout_labels = {}
        for dt in unique_breakouts:
            pats_for_date = [p for p in ticker_patterns if pd.to_datetime(p['breakout_date']).strftime('%Y-%m-%d') == dt]
            algo_status_list = []
            for p in pats_for_date:
                short_algo = p['algo'].replace(" Peak Prominence", "").replace(" Window Extremum", "").replace(" Pattern Matching", "")
                status_txt = "Sukses" if p['success'] == True else ("Gagal" if p['success'] == False else "Aktif")
                
                # Get accuracy percentage for this algorithm on this stock
                if p['algo'] == "Scipy Peak Prominence":
                    acc = stats_scipy['akurasi']
                elif p['algo'] == "Rolling Window Extremum":
                    acc = stats_rolling['akurasi']
                else:
                    acc = stats_zigzag['akurasi']
                    
                algo_status_list.append(f"{short_algo}: {status_txt} ({acc:.0f}%)")
            breakout_labels[dt] = f"{dt} (" + " | ".join(algo_status_list) + ")"
            
        if len(unique_breakouts) > 1:
            selected_breakout_str = st.selectbox(
                "Ditemukan beberapa tanggal pola terdeteksi, pilih tanggal breakout untuk dibandingkan:",
                unique_breakouts,
                format_func=lambda x: breakout_labels[x]
            )
        elif len(unique_breakouts) == 1:
            selected_breakout_str = unique_breakouts[0]
            st.markdown(f"**Tanggal Breakout Pola:** {breakout_labels[selected_breakout_str]}")
        else:
            selected_breakout_str = None
            
        # Filter patterns for the selected breakout date
        pattern_scipy = None
        pattern_rolling = None
        pattern_zigzag = None
        
        if selected_breakout_str:
            pats_scipy_match = [p for p in pats_scipy if pd.to_datetime(p['breakout_date']).strftime('%Y-%m-%d') == selected_breakout_str]
            if pats_scipy_match:
                pattern_scipy = pats_scipy_match[0]
                
            pats_rolling_match = [p for p in pats_rolling if pd.to_datetime(p['breakout_date']).strftime('%Y-%m-%d') == selected_breakout_str]
            if pats_rolling_match:
                pattern_rolling = pats_rolling_match[0]
                
            pats_zigzag_match = [p for p in pats_zigzag if pd.to_datetime(p['breakout_date']).strftime('%Y-%m-%d') == selected_breakout_str]
            if pats_zigzag_match:
                pattern_zigzag = pats_zigzag_match[0]
        
        # 1. Scipy Peak Prominence Section
        render_algorithm_chart("Scipy Peak Prominence", pattern_scipy, df_ticker, "scipy")
        st.markdown("---")
        
        # 2. Rolling Window Extremum Section
        render_algorithm_chart("Rolling Window Extremum", pattern_rolling, df_ticker, "rolling")
        st.markdown("---")
        
        # 3. ZigZag Pattern Matching Section
        render_algorithm_chart("ZigZag Pattern Matching", pattern_zigzag, df_ticker, "zigzag")
        
        # Perbandingan Ketiga Algoritma untuk Emiten Terpilih
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"Perbandingan Performa Ketiga Algoritma Khusus Saham {selected_emiten}")
        
        df_compare = pd.DataFrame([
            {
                "Algoritma Deteksi": "Scipy Peak Prominence",
                "Total Pola": stats_scipy['total'],
                "Aktif": stats_scipy['aktif'],
                "Sukses (TP)": stats_scipy['sukses'],
                "Gagal (SL)": stats_scipy['gagal'],
                "Akurasi Sinyal Jual": f"{stats_scipy['akurasi']:.2f}%" if stats_scipy['total'] - stats_scipy['aktif'] > 0 else "0.00% (Belum teruji)"
            },
            {
                "Algoritma Deteksi": "Rolling Window Extremum",
                "Total Pola": stats_rolling['total'],
                "Aktif": stats_rolling['aktif'],
                "Sukses (TP)": stats_rolling['sukses'],
                "Gagal (SL)": stats_rolling['gagal'],
                "Akurasi Sinyal Jual": f"{stats_rolling['akurasi']:.2f}%" if stats_rolling['total'] - stats_rolling['aktif'] > 0 else "0.00% (Belum teruji)"
            },
            {
                "Algoritma Deteksi": "ZigZag Pattern Matching",
                "Total Pola": stats_zigzag['total'],
                "Aktif": stats_zigzag['aktif'],
                "Sukses (TP)": stats_zigzag['sukses'],
                "Gagal (SL)": stats_zigzag['gagal'],
                "Akurasi Sinyal Jual": f"{stats_zigzag['akurasi']:.2f}%" if stats_zigzag['total'] - stats_zigzag['aktif'] > 0 else "0.00% (Belum teruji)"
            }
        ])
        st.dataframe(df_compare.reset_index(drop=True), use_container_width=True)

# TAB 2: DATA HASIL PREDIKSI
with tab2:
    if not all_detected:
        st.info("Tidak ada hasil prediksi terdeteksi.")
    else:
        df_all = pd.DataFrame(all_detected)
        
        st.subheader("Sinyal Prediksi Aktif (Potensi Masa Depan)")
        df_active_table = df_all[df_all['success'].isna()].copy()
        if len(df_active_table) > 0:
            df_active_tampil = df_active_table[['ticker', 'algo', 'breakout_date', 'breakout_price', 'target_price', 'stop_loss_price']].copy()
            df_active_tampil['breakout_date'] = df_active_tampil['breakout_date'].dt.strftime('%Y-%m-%d')
            df_active_tampil.columns = ['Kode Saham', 'Algoritma', 'Tgl Breakout', 'Harga Breakout', 'Target Harga (TP)', 'Batas Rugi (SL)']
            st.dataframe(df_active_tampil.reset_index(drop=True), use_container_width=True)
        else:
            st.write("Tidak ada pola aktif berjalan saat ini di pasar.")
            
        st.markdown("---")
        
        st.subheader("Sinyal Sukses Teruji (Take Profit)")
        df_success_table = df_all[df_all['success'] == True].copy()
        if len(df_success_table) > 0:
            df_success_tampil = df_success_table[['ticker', 'algo', 'breakout_date', 'outcome_date', 'breakout_price', 'target_price']].copy()
            df_success_tampil['breakout_date'] = df_success_tampil['breakout_date'].dt.strftime('%Y-%m-%d')
            df_success_tampil['outcome_date'] = df_success_tampil['outcome_date'].dt.strftime('%Y-%m-%d')
            df_success_tampil.columns = ['Kode Saham', 'Algoritma', 'Tgl Breakout', 'Tgl Target Tercapai', 'Harga Breakout', 'Target Terpenuhi (TP)']
            st.dataframe(df_success_tampil.reset_index(drop=True), use_container_width=True)
        else:
            st.write("Belum ada emiten yang teruji sukses.")

