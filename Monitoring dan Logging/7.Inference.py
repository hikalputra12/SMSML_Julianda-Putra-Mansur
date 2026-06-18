import os
import time
import random
import requests
import pandas as pd

def main():
    # Definisikan path ke dataset bersih secara dinamis
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(
        base_dir, 
        "Eksperimen_SML_Julianda-Putra-Mansur", 
        "preprocessing", 
        "loan_approval_preprocessing", 
        "dataset_clean.csv"
    )
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset bersih tidak ditemukan di {data_path}!")
        return

    print(f"[INFO] Membaca dataset dari {data_path}...")
    df = pd.read_csv(data_path)
    
    # Pisahkan target column jika ada
    if 'loan_status' in df.columns:
        df = df.drop(columns=['loan_status'])
        
    columns = list(df.columns)
    records = df.to_dict(orient="records")
    
    proxy_url = "http://127.0.0.1:8000/predict"
    print(f"[INFO] Mulai mengirim request simulasi ke {proxy_url}...")
    print("[INFO] Tekan Ctrl+C untuk menghentikan simulasi.")
    
    count = 0
    try:
        while True:
            # Ambil record acak dari dataset
            record = random.choice(records)
            
            # Format payload MLflow menggunakan split format
            payload = {
                "dataframe_split": {
                    "columns": columns,
                    "data": [[record[col] for col in columns]]
                }
            }
            
            # Tambahkan kemungkinan acak untuk mengirim data salah guna men-generate error rate
            # (Misal 2% dari total request akan dikirim dengan payload kosong)
            if random.random() < 0.02:
                payload_send = {}
            else:
                payload_send = payload
                
            start_time = time.time()
            try:
                response = requests.post(proxy_url, json=payload_send, timeout=3)
                latency = time.time() - start_time
                
                count += 1
                if response.status_code == 200:
                    print(f"[{count}] Request Sukses | Latency: {latency:.4f}s | Prediksi: {response.json()}")
                else:
                    print(f"[{count}] Request Gagal (Expected 2%) | Status: {response.status_code} | Respon: {response.text}")
            except Exception as e:
                print(f"[{count}] Koneksi Error ke Proxy Exporter: {e}")
                
            # Tunggu interval acak antara 0.5s - 2.5s agar pola visualisasi Grafana terlihat natural
            time.sleep(random.uniform(0.5, 2.5))
            
    except KeyboardInterrupt:
        print("\n[INFO] Simulasi dihentikan oleh pengguna.")

if __name__ == "__main__":
    main()
