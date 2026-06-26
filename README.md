# Dashboard Bursa Cerdas - Deteksi & Backtesting Pola Double Top Saham

Sistem cerdas berbasis aturan (*rule-based heuristic*) untuk mendeteksi dan menguji kembali (*backtesting*) pola pembalikan arah turun (*bearish reversal*) **Double Top** pada data transaksi harian saham. Aplikasi ini membandingkan tiga algoritma sekaligus secara interaktif:
1. **SciPy Peak Prominence** (Pendekatan *Signal Processing*)
2. **Rolling Window Extremum** (Pendekatan *Quantitative Finance*)
3. **ZigZag Pattern Matching** (Pendekatan *Technical Analysis*)

---

## 🛠️ Cara Deploy & Menjalankan Aplikasi di Localhost

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi di komputer Anda secara lokal:

### 1. Prasyarat (*Prerequisites*)
Pastikan komputer Anda sudah terinstal **Python** (versi 3.8 atau yang lebih baru, direkomendasikan Python 3.11).

### 2. Instalasi Library / Dependensi
Buka Terminal (Mac/Linux) atau Command Prompt / PowerShell (Windows), kemudian jalankan perintah berikut untuk menginstal seluruh pustaka Python yang dibutuhkan:

```bash
pip install streamlit pandas numpy scipy plotly
```

### 3. Persiapkan Dataset
Pastikan berkas dataset transaksi saham berikut berada di **direktori/folder yang sama** dengan file `app.py`:
* Berkas data: `transaksi_harian_202606130928.csv`

### 4. Menjalankan Aplikasi Streamlit
Dalam terminal, pastikan posisi direktori kerja Anda berada di folder yang berisi file `app.py`. Kemudian jalankan perintah berikut:

```bash
streamlit run app.py
```

### 5. Mengakses Dashboard
Setelah perintah di atas dijalankan, aplikasi akan otomatis membuka browser internet Anda. Jika tidak terbuka otomatis, salin dan buka alamat berikut di browser Anda:

* URL Utama: `http://localhost:8501`

---

## 📊 Fitur Utama Dashboard
* **Filter Emiten Berdasarkan Status Pola:** Memilih antara "Semua Emiten" atau menyaring "Hanya Sinyal Aktif (Ongoing)" untuk memotong risiko volatilitas dan membantu analisis *short-selling*.
* **Dropdown Penyelarasan Tanggal:** Memungkinkan analisis komparatif dari ketiga algoritma pada garis waktu *breakout* yang persis sama.
* **Grafik Candlestick Interaktif:** Menggunakan Plotly untuk visualisasi lilin tanpa celah akhir pekan (*holiday/weekend gaps*) demi estetika bursa profesional.
* **Warna Status Emas:** TP (Take Profit) Sukses ditandai dengan warna emas (`#ffd700`) untuk visualisasi premium.
* **Tabel Perbandingan Performa:** Menyajikan data statistik akurasi, total TP, dan total SL dari masing-masing algoritma di bagian bawah dashboard untuk analisis kuantitatif.
