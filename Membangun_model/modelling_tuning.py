import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skopt import BayesSearchCV

def main():
    # Inisialisasi DagsHub tracking remote
    dagshub.init(repo_owner='hikalputra12', repo_name='Eksperimen_SML_Julianda-Putra-Mansur', mlflow=True)

    # Mengambil dataset bersih secara dinamis
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "Eksperimen_SML_Julianda-Putra-Mansur", "preprocessing", "loan_approval_preprocessing", "dataset_clean.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data bersih tidak ditemukan di {data_path}. Jalankan kriteria 1 terlebih dahulu!")
        
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi Eksperimen MLflow Tuning
    mlflow.set_experiment("Loan_Approval_Tuning")
    
    with mlflow.start_run(run_name="RandomForest_BayesianSearch"):
        # Ruang pencarian Hyperparameter untuk Bayesian Optimization
        search_spaces = {
            'n_estimators': (50, 150),
            'max_depth': (5, 15),
            'min_samples_split': (2, 10)
        }
        
        rf = RandomForestClassifier(random_state=42)
        # Menggunakan BayesSearchCV untuk optimasi Bayesian
        opt = BayesSearchCV(
            estimator=rf,
            search_spaces=search_spaces,
            n_iter=10, # jumlah iterasi pencarian
            cv=3,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        opt.fit(X_train, y_train)
        
        best_model = opt.best_estimator_
        best_params = opt.best_params_
        
        # Logging parameter terbaik hasil pencarian secara manual
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
            
        # Evaluasi model terbaik hasil tuning
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Logging metrik hasil tuning secara manual ke DagsHub
        mlflow.log_metric("best_accuracy", acc)
        mlflow.log_metric("best_precision", prec)
        mlflow.log_metric("best_recall", rec)
        mlflow.log_metric("best_f1_score", f1)
        
        # Logging model tersetel terbaik sebagai artefak resmi
        mlflow.sklearn.log_model(best_model, "tuned_best_model")
        
        print("="*50)
        print("[+] Hyperparameter Tuning (Bayesian Optimization) Selesai & Di-log!")
        print(f"Best Params  : {dict(best_params)}")
        print(f"Best Accuracy: {acc:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()