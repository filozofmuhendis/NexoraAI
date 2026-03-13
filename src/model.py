import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import pickle
import os

def load_data(file_path):
    """
    Final veri setini yükler ve X, y olarak döner.
    """
    if not os.path.exists(file_path):
        print(f"[!] Veri seti bulunamadı: {file_path}. Önce pipeline'ı çalıştırın.")
        return None, None
    
    df = pd.read_csv(file_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    print(f"[*] Veri yüklendi: {X.shape[0]} Örnek, {X.shape[1]} Öznitelik.")
    return X, y

def train_xgboost_baseline(X, y):
    """
    XGBoost modelini bir stratifiye train-test split ile eğitir.
    Ardından 5-fold cross validation ile performans ölçümü yapar.
    """
    # 1. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("[*] XGBoost Baseline Eğitimi Başlıyor...")
    
    # XGBoost Parameters (Baseline)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    # 2. 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_train, y_train, cv=skf, 
        scoring=['accuracy', 'f1', 'roc_auc'],
        return_train_score=False
    )
    
    print("\n--- 5-Fold Cross Validation Sonuçları (Eğitim Seti Üzerinde) ---")
    print(f"Ortalama Accuracy: {cv_results['test_accuracy'].mean():.4f}")
    print(f"Ortalama F1-Score: {cv_results['test_f1'].mean():.4f}")
    print(f"Ortalama ROC-AUC:  {cv_results['test_roc_auc'].mean():.4f}")
    print("-----------------------------------------------------------\n")

    # 3. Final Model Eğitimi (X_train üzerinde)
    model.fit(X_train, y_train)
    
    # 4. Test Seti Performansı
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("--- Test Seti Performansı ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Skoru: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Feature Importance (Top 10)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nEn Önemli 10 Öznitelik:")
    print(importance.head(10))
    
    # Modeli Kaydet
    with open('baseline_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\n[+] Model 'baseline_model.pkl' olarak kaydedildi.")

def main():
    X, y = load_data('final_variant_dataset.csv')
    if X is not None:
        train_xgboost_baseline(X, y)

if __name__ == "__main__":
    main()
