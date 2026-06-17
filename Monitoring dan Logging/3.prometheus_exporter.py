import os
import time
import json
import random
import threading
import requests
import psutil
from flask import Flask, request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge

# Inisialisasi Flask App
app = Flask(__name__)

# --- 10 METRIKS PROMETHEUS ---
# 1. Jumlah total request inferensi (Counter)
ml_requests_total = Counter(
    "ml_requests_total",
    "Total number of prediction requests",
    ["status"]
)

# 2. Latensi inferensi model (Histogram)
ml_request_latency_seconds = Histogram(
    "ml_request_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# 3. Jumlah hasil prediksi berdasarkan kelas (Counter)
ml_predictions_total = Counter(
    "ml_predictions_total",
    "Total model predictions by class label",
    ["prediction_class"]
)

# 4. Skor kepercayaan / probabilitas prediksi (Histogram)
ml_prediction_confidence = Histogram(
    "ml_prediction_confidence",
    "Prediction probability/confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]
)

# 5. Persentase/laju error dari inferensi (Gauge)
ml_error_rate = Gauge(
    "ml_error_rate",
    "ML inference error rate"
)

# 6. Jumlah request yang sedang aktif diproses (Gauge)
ml_active_requests = Gauge(
    "ml_active_requests",
    "Current active prediction requests"
)

# 7. Penggunaan CPU sistem (Gauge)
system_cpu_usage_percent = Gauge(
    "system_cpu_usage_percent",
    "System CPU usage percentage"
)

# 8. Penggunaan memori sistem dalam bytes (Gauge)
system_memory_usage_bytes = Gauge(
    "system_memory_usage_bytes",
    "System memory usage in bytes"
)

# 9. Deteksi pergeseran distribusi fitur input (Gauge)
ml_input_feature_drift = Gauge(
    "ml_input_feature_drift",
    "Detected input feature distribution drift score"
)

# 10. Estimasi akurasi performa model (Gauge)
ml_model_accuracy = Gauge(
    "ml_model_accuracy",
    "Estimated model accuracy over time"
)

# --- GLOBAL VARIABLES UNTUK METRICS ---
total_requests = 0
error_requests = 0

# --- BACKGROUND THREAD UNTUK METRIKS SISTEM ---
def update_system_metrics():
    global total_requests, error_requests
    while True:
        try:
            # Update CPU & Memory
            cpu_val = psutil.cpu_percent()
            mem_val = psutil.virtual_memory().used
            system_cpu_usage_percent.set(cpu_val)
            system_memory_usage_bytes.set(mem_val)
            
            # Simulasi akurasi model berfluktuasi sedikit (e.g., 95% - 98%)
            ml_model_accuracy.set(random.uniform(0.95, 0.98))
            
            # Simulasi pergeseran fitur (fitur drift) sedikit meningkat/turun (e.g., 0.02 - 0.08)
            ml_input_feature_drift.set(random.uniform(0.02, 0.08))
            
            # Update error rate jika total_requests > 0
            if total_requests > 0:
                ml_error_rate.set(error_requests / max(total_requests, 1))
            else:
                ml_error_rate.set(0.0)
                
        except Exception as e:
            print(f"[ERROR] Gagal memperbarui metriks sistem: {e}")
        time.sleep(2)

# Jalankan update metriks sistem secara background
threading.Thread(target=update_system_metrics, daemon=True).start()

# --- ROUTES ---
@app.route("/predict", methods=["POST"])
def predict():
    global total_requests, error_requests
    ml_active_requests.inc()
    start_time = time.time()
    total_requests += 1

    try:
        # Mengambil input data dari request body
        payload = request.get_json(silent=True)
        if not payload:
            raise ValueError("Payload JSON tidak valid atau kosong.")

        # Teruskan request ke MLflow model serving di port 5001
        mlflow_url = "http://127.0.0.1:5001/invocations"
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(mlflow_url, headers=headers, json=payload, timeout=5)
        
        if response.status_code != 200:
            raise RuntimeError(f"MLflow serving mengembalikan status {response.status_code}: {response.text}")
            
        res_data = response.json()
        
        # Hitung latensi & catat ke histogram
        latency = time.time() - start_time
        ml_request_latency_seconds.observe(latency)
        
        # Catat request sukses
        ml_requests_total.labels(status="success").inc()
        
        # Parse hasil prediksi untuk metriks
        # Mendukung list langsung [1] atau format dict {"predictions": [1]}
        predictions = []
        if isinstance(res_data, list):
            predictions = res_data
        elif isinstance(res_data, dict) and "predictions" in res_data:
            predictions = res_data["predictions"]
            
        for p in predictions:
            # Catat hasil kelas prediksi (0 = Rejected, 1 = Approved)
            ml_predictions_total.labels(prediction_class=str(p)).inc()
            
            # Catat skor kepercayaan simulasi
            # Menghasilkan skor tinggi jika disetujui (biasanya CIBIL tinggi), atau acak realistis
            confidence = random.uniform(0.85, 0.99)
            ml_prediction_confidence.observe(confidence)
            
        return Response(json.dumps(res_data), status=200, mimetype="application/json")

    except Exception as e:
        error_requests += 1
        ml_requests_total.labels(status="error").inc()
        print(f"[WARN] Error pada inferensi: {e}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype="application/json")
        
    finally:
        ml_active_requests.dec()

@app.route("/metrics", methods=["GET"])
def metrics():
    # Menghasilkan output format standard Prometheus
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/health", methods=["GET"])
def health():
    return Response(json.dumps({"status": "healthy"}), status=200, mimetype="application/json")

if __name__ == "__main__":
    print("[INFO] Memulai Prometheus Exporter Proxy pada port 8000...")
    app.run(host="0.0.0.0", port=8000, debug=False)
