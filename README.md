# Eksperimen Siklus Hidup Machine Learning (SML) - Sistem Prediksi Loan Approval

Proyek ini merupakan implementasi terintegrasi dari seluruh siklus hidup pengembangan model Machine Learning (MLOps). Cakupan proyek meliputi pembersihan data (preprocessing), pelatihan model dasar, hyperparameter tuning dengan pelacakan eksperimen (**MLflow** & **DagsHub**), pengemasan model (**MLProject** & **Docker Hub**), hingga pemantauan performa model secara real-time di lingkungan produksi menggunakan **Prometheus** dan **Grafana**.

Proyek ini dirancang untuk memenuhi kriteria evaluasi tingkat **Advance** pada Kriteria 4 Dicoding.

---

## 📂 Struktur Direktori Proyek

Berikut adalah tata letak folder dan file utama dalam repositori ini beserta penjelasannya:

```text
├── .gitignore                                 # Konfigurasi pengabaian file cache, virtualenv, dan data lokal
├── README.md                                  # Dokumentasi utama proyek (file ini)
│
├── Eksperimen_SML_Julianda-Putra-Mansur/      # Data & Preprocessing
│   ├── loan_approval_raw/                     # Folder berisi dataset mentah (dataset_raw.csv)
│   └── preprocessing/                         # Folder berisi hasil preprocessing dan data bersih (dataset_clean.csv)
│
├── Membangun_model/                           # Eksperimen Pelatihan & Tuning Model
│   ├── modelling.py                           # Kode dasar pelatihan model RandomForest
│   ├── modelling_tuning.py                    # Kode tuning hyperparameter model terbaik (RandomForest)
│   ├── requirements.txt                       # Dependensi library Python untuk pemodelan
│   ├── DagsHub.txt                            # Tautan repositori DagsHub untuk tracking MLflow remote
│   └── screenshoot_...                        # Screenshot bukti eksperimen & tracking model di MLflow
│
├── MLProject/                                 # Pengemasan Model (MLflow Project)
│   ├── MLProject                              # Konfigurasi entrypoint proyek MLflow
│   ├── conda.yaml                             # Spesifikasi environment conda untuk reproduksibilitas
│   ├── modelling.py                           # Skrip pelatihan RandomForest terintegrasi MLflow
│   └── docker_hub_link.txt                    # Tautan Docker Image model yang telah dipublikasikan ke Docker Hub
│
└── Monitoring dan Logging/                    # Infrastruktur Monitoring & Alerting
    ├── 2.prometheus.yml                       # Konfigurasi scraping target untuk server Prometheus
    ├── 3.prometheus_exporter.py               # Flask Proxy Exporter (menjembatani request, menghitung metrik & MLflow serving)
    ├── 7.inference.py                         # Simulator client pengirim request inferensi acak secara berkala
    ├── README_Monitoring.md                   # Panduan lengkap langkah demi langkah setup monitoring & alerting
    │
    # --- Folder Bukti Pengumpulan (Screenshots) ---
    ├── 1.bukti_serving/                       # Screenshot bukti MLflow serving berjalan di port 5001
    ├── 4.bukti monitoring Prometheus/          # Screenshot dashboard Prometheus (port 9090) & targets UP
    ├── 5.bukti monitoring Grafana/            # Screenshot dashboard Grafana dengan 10 panel metrik lengkap
    └── 6.bukti alerting Grafana/              # Screenshot 3 alerting rules (kondisi normal & firing)
```

---

## 🛠️ Alur Kerja & Cara Menjalankan Proyek

Seluruh komponen monitoring saat ini sudah diintegrasikan dan dapat dijalankan di Windows dengan langkah-langkah berikut:

### Langkah 1: Pelatihan & Tuning Model (Opsional)
Untuk melatih model RandomForest dan melacak hasil metrik performanya secara remote di DagsHub:
```powershell
python "Membangun_model/modelling_tuning.py"
```

### Langkah 2: Menjalankan MLflow Model Serving
Menyajikan model terbaik hasil pelatihan pada port `5001`:
```powershell
mlflow models serve -m "mlruns/599045318861770893/b0a3ab890c0c4117b9e8085fb48801a1/artifacts/tuned_best_model" -p 5001 --env-manager local
```

### Langkah 3: Menjalankan Prometheus Exporter Proxy
Jalankan proxy Flask di port `8000` untuk merekam metrik inferensi, penggunaan sistem, serta meneruskannya ke port serving `5001`:
```powershell
python "Monitoring dan Logging/3.prometheus_exporter.py"
```

### Langkah 4: Menjalankan Prometheus & Grafana via Docker Desktop
Pastikan **Docker Desktop** Anda sudah aktif (status *Engine Running*), lalu jalankan perintah ini di PowerShell untuk memulai server monitoring:

**Prometheus Server (Port 9090):**
```powershell
docker run -d --name prometheus -p 9090:9090 -v "c:/Users/Julianda/Eksperimen_SML_Julianda-Putra-Mansur/Monitoring dan Logging/2.prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus
```

**Grafana Server (Port 3000):**
```powershell
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

### Langkah 5: Jalankan Simulasi Request Client
Jalankan generator beban inferensi secara berkelanjutan agar metrik terisi secara dinamis di Prometheus & Grafana:
```powershell
python "Monitoring dan Logging/7.inference.py"
```

---

## 📊 Metrik & Alerting yang Dipantau

### 10 Metriks Utama Grafana
* **Total Prediction Requests:** Jumlah total request (sukses + error).
* **Average Inference Latency:** Rata-rata waktu pemrosesan prediksi (detik).
* **Predictions by Class:** Proporsi hasil klasifikasi (Approved vs Rejected).
* **95th Percentile Confidence Score:** Estimasi confidence score pada persentil 95.
* **System CPU Usage:** Persentase penggunaan CPU sistem host.
* **System Memory Usage:** Memori RAM sistem yang terpakai (bytes).
* **ML Inference Error Rate:** Tingkat kegagalan request inferensi.
* **Active In-Flight Requests:** Jumlah request yang sedang diproses secara paralel.
* **Input Feature Drift Score:** Skor deteksi pergeseran distribusi data input.
* **Estimated Model Accuracy:** Estimasi tingkat akurasi performa model.

### 3 Alerting Rules yang Dikonfigurasi
1. **High Latency Alert:** Memicu peringatan jika latensi inferensi > 0.05 detik.
2. **High Error Rate Alert:** Memicu peringatan jika tingkat kegagalan request > 5%.
3. **High CPU Usage Alert:** Memicu peringatan jika penggunaan CPU sistem > 10% (untuk simulasi lokal) / 80%.

Semua bukti visual berupa screenshot konfigurasi rules, status firing, dan visualisasi grafik dapat ditemukan pada folder **`bukti`** masing-masing komponen di dalam direktori `Monitoring dan Logging`.