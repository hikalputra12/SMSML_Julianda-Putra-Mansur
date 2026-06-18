# Panduan Menjalankan Prometheus & Grafana (Kriteria 4 - Advance: 4 Poin)

Dokumen ini berisi panduan lengkap langkah demi langkah untuk menjalankan model serving, Prometheus exporter, Prometheus Server, Grafana, serta konfigurasi dashboard dan alerting guna mendapatkan nilai sempurna (**4 poin - Advance**) pada Kriteria 4 Dicoding.

---

## 🛠️ Persiapan Environment & Dependensi

Pastikan semua library Python yang dibutuhkan sudah terinstal di environment Anda. Jalankan perintah berikut di terminal/Powershell:

```powershell
pip install flask requests psutil prometheus-client mlflow pandas scikit-learn
```

---

## 🚀 Langkah 1: Menjalankan MLflow Model Serving
Untuk memenuhi syarat serving model pada local environment, kita akan men-serve model hasil training yang tersimpan di folder `mlruns`. 

Di proyek Anda, model terbaik yang sudah di-tuning berada di path:
`mlruns/599045318861770893/b0a3ab890c0c4117b9e8085fb48801a1/artifacts/tuned_best_model`

Jalankan perintah berikut di terminal untuk melakukan serving model pada port `5001`:

```powershell
mlflow models serve -m "mlruns/599045318861770893/b0a3ab890c0c4117b9e8085fb48801a1/artifacts/tuned_best_model" -p 5001 --env-manager local
```
> **Catatan:** Parameter `--env-manager local` digunakan agar MLflow menggunakan package Python yang sudah terpasang di sistem tanpa harus mengunduh dan membuat environment Conda baru (mempercepat proses running).

---

## 🔌 Langkah 2: Menjalankan Prometheus Exporter Proxy
Buka **terminal/tab baru**, lalu jalankan script proxy exporter yang berfungsi menjembatani request inferensi, menghitung metrics, dan meneruskannya ke MLflow serving:

```powershell
python "Monitoring dan Logging/3.prometheus_exporter.py"
```
Proxy exporter ini akan berjalan pada port `8000`. Anda dapat memverifikasi metrics yang dihasilkan dengan membuka [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics) di browser Anda.

---

## ⏱️ Langkah 3: Menjalankan Prometheus Server

Ada dua pilihan untuk menjalankan Prometheus Server:

### Opsi A: Menggunakan Docker (Sangat Direkomendasikan)
Jika menggunakan Docker, harap dicatat bahwa container Docker tidak dapat mengakses `127.0.0.1:8000` secara langsung karena localhost merujuk ke container itu sendiri.

1. Buka file `Monitoring dan Logging/2.prometheus.yml`.
2. Ubah target dari `"127.0.0.1:8000"` menjadi `"host.docker.internal:8000"`.
3. Jalankan container Prometheus melalui PowerShell:
   ```powershell
   docker run -d --name prometheus -p 9090:9090 -v "${PWD}/Monitoring dan Logging/2.prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus
   ```

### Opsi B: Menjalankan secara Lokal (Tanpa Docker)
1. Unduh binary Prometheus untuk Windows di [Situs Resmi Prometheus](https://prometheus.io/download/).
2. Ekstrak file `.zip` hasil unduhan.
3. Masuk ke folder hasil ekstrak, lalu jalankan Prometheus dengan memuat file konfigurasi dari workspace:
   ```powershell
   .\prometheus.exe --config.file="C:\path\to\your\workspace\Monitoring dan Logging\2.prometheus.yml"
   ```
4. Akses dashboard Prometheus di [http://localhost:9090](http://localhost:9090).

---

## 📊 Langkah 4: Menjalankan Grafana

Jalankan Grafana menggunakan Docker (cara tercepat di Windows):

```powershell
docker run -d --name grafana -p 3000:3000 grafana/grafana
```
*Atau, jika Anda tidak menggunakan Docker, silakan unduh dan jalankan Grafana standalone installer untuk Windows.*

Akses dashboard Grafana di [http://localhost:3000](http://localhost:3000) (Username default: `admin`, Password default: `admin`). Anda akan diminta membuat password baru setelah login pertama kali.

---

## 🔄 Langkah 5: Menjalankan Simulasi Request (Inference Client)
Buka **terminal/tab baru**, lalu jalankan script load generator untuk mulai mengirimkan request prediksi acak dari dataset bersih secara berkelanjutan. Ini akan menyuplai data ke Prometheus & Grafana secara real-time:

```powershell
python "Monitoring dan Logging/7.inference.py"
```

---

## 📈 Langkah 6: Konfigurasi Dashboard di Grafana (Minimal 10 Metriks)

Untuk mendapatkan nilai sempurna (**Advance: minimal 10 metriks**), lakukan langkah berikut:

### 1. Tambahkan Prometheus sebagai Data Source
1. Di Grafana, buka menu sidebar kiri -> **Connections** -> **Data sources**.
2. Klik **Add data source**, pilih **Prometheus**.
3. Di bagian **Connection URL**, masukkan:
   - `http://host.docker.internal:9090` (Jika Grafana berada di dalam Docker)
   - `http://localhost:9090` (Jika Grafana & Prometheus berjalan di local OS langsung)
4. Scroll ke bawah dan klik **Save & test**. Pastikan muncul pesan sukses hijau (*"Data source is working"*).

### 2. Buat Dashboard Baru
1. Buka sidebar kiri -> **Dashboards** -> klik **New** -> **New Dashboard**.
2. Sebelum menambahkan panel, klik **Dashboard settings** (ikon roda gigi di kanan atas).
3. **Penting (Kredensial Akun Dicoding):** Ubah nama Dashboard dengan format nama username Dicoding Anda, contoh: `Loan-Monitoring-<Username_Dicoding_Anda>`.
4. Klik **Save dashboard**.

### 3. Tambahkan 10 Panel Metriks Berbeda
Tambahkan panel visualisasi baru untuk masing-masing dari 10 metriks berikut menggunakan query PromQL yang sesuai:

| No | Nama Panel / Metriks | Tipe Visualisasi | Query PromQL | Deskripsi |
|---|---|---|---|---|
| 1 | **Total Prediction Requests** | Time Series / Stat | `sum(ml_requests_total)` | Jumlah total request prediksi (sukses + error) |
| 2 | **Average Inference Latency** | Time Series | `rate(ml_request_latency_seconds_sum[1m]) / rate(ml_request_latency_seconds_count[1m])` | Rata-rata latensi inferensi model (dalam detik) |
| 3 | **Predictions by Class** | Pie Chart / Bar Gauge | `sum by (prediction_class) (ml_predictions_total)` | Total prediksi berdasarkan kelas (0 = Rejected, 1 = Approved) |
| 4 | **95th Percentile Confidence Score**| Gauge / Stat | `histogram_quantile(0.95, sum(rate(ml_prediction_confidence_bucket[5m])) by (le))` | Estimasi nilai keyakinan (confidence) prediksi pada persentil ke-95 |
| 5 | **System CPU Usage** | Gauge | `system_cpu_usage_percent` | Persentase penggunaan CPU sistem secara real-time |
| 6 | **System Memory Usage** | Time Series | `system_memory_usage_bytes` | Jumlah penggunaan memori RAM sistem dalam bytes |
| 7 | **ML Inference Error Rate** | Time Series / Stat | `ml_error_rate` | Rasio request yang gagal/error terhadap total request |
| 8 | **Active In-Flight Requests** | Stat / Time Series | `ml_active_requests` | Jumlah request prediksi yang sedang aktif diproses saat ini |
| 9 | **Input Feature Drift Score** | Time Series | `ml_input_feature_drift` | Skor deteksi pergeseran fitur input (data drift score) |
| 10| **Estimated Model Accuracy** | Stat / Gauge | `ml_model_accuracy` | Estimasi tingkat akurasi performa model seiring waktu |

---

## 🔔 Langkah 7: Membuat 3 Alerting Rules di Grafana

Kriteria Advance mewajibkan Anda untuk membuat **minimal 3 alerting rules** menggunakan Grafana. Ikuti langkah ini untuk membuatnya:

1. Di Grafana, masuk ke menu sidebar kiri -> **Alerting** -> **Alert rules** -> Klik **Create alert rule**.
2. Buat **3 Alert Rule** terpisah dengan ketentuan sebagai berikut:

### Alert 1: High Latency Alert (Deteksi Latensi Lambat)
- **Rule Name:** `High Latency Alert - <Username_Dicoding>`
- **Set Query (A):** Pilih Data Source Prometheus, isi query dengan:
  ```promql
  rate(ml_request_latency_seconds_sum[1m]) / rate(ml_request_latency_seconds_count[1m])
  ```
- **Threshold (C):** Atur kondisi alert bila nilai rata-rata latensi di atas **0.5** (atau sesuaikan ke **0.05** agar alert terpicu dengan cepat saat pengujian).

### Alert 2: High Error Rate Alert (Deteksi Banyak Error)
- **Rule Name:** `High Error Rate Alert - <Username_Dicoding>`
- **Set Query (A):** Pilih Data Source Prometheus, isi query dengan:
  ```promql
  ml_error_rate
  ```
- **Threshold (C):** Atur kondisi alert bila error rate di atas **0.05** (artinya tingkat kegagalan > 5%). Anda bisa memicunya dengan sengaja karena script `7.inference.py` memiliki peluang 2% mengirim payload kosong yang menghasilkan error.

### Alert 3: High CPU Usage Alert (Deteksi Beban Server Tinggi)
- **Rule Name:** `High CPU Usage Alert - <Username_Dicoding>`
- **Set Query (A):** Pilih Data Source Prometheus, isi query dengan:
  ```promql
  system_cpu_usage_percent
  ```
- **Threshold (C):** Atur kondisi alert jika CPU usage di atas **80** (atau sesuaikan ke **10** agar mudah terpicu saat melakukan uji coba local).

---

## 📸 Panduan Screenshot untuk Pengumpulan Dicoding

Pastikan tangkapan layar (screenshot) yang Anda ambil memenuhi syarat berikut:
1. **Bukti Serving Model:** Screenshot terminal saat MLflow model serve sedang berjalan lancar di port 5001.
2. **Bukti Monitoring Prometheus:** Screenshot dashboard Prometheus (di `localhost:9090`) yang menampilkan query metric (misal: `ml_requests_total`) dengan status sukses scraping.
3. **Bukti Monitoring Grafana (PENTING):** Screenshot dashboard Grafana Anda yang menampilkan **ke-10 panel visualisasi** secara jelas, serta menunjukkan **Nama Dashboard** yang memuat **Username Akun Dicoding Anda**.
4. **Bukti Alerting Grafana:** Screenshot menu **Alerting > Alert rules** di Grafana yang menampilkan status **3 alert** yang telah Anda buat di atas.
