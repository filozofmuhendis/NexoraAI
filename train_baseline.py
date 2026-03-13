from src.model import main as run_training
import os

if __name__ == "__main__":
    print("--- XGBoost Baseline Model Eğitimi ---")
    
    if not os.path.exists('final_variant_dataset.csv'):
        print("[!] Veri seti bulunamadı. Önce 'build_dataset.py' dosyasını çalıştırın.")
    else:
        run_training()
