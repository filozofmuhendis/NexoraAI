import pandas as pd
import numpy as np
import os
import argparse
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, auc, f1_score, balanced_accuracy_score, classification_report
)
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"[!] Hata: {file_path} bulunamadı.")
        return None, None
        
    df = pd.read_csv(file_path)
    
    # Veri sızıntısını önleyen drop (ID, CHROM, POS vb)
    leakage_cols = ['ID', 'CHROM', 'POS', '#CHROM', 'REF', 'ALT', 'variant_key', 'hgvsp', 'AA_REF', 'AA_ALT']
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors='ignore')
    
    if 'target' not in df.columns:
        raise ValueError("[!] Veri setinde 'target' kolonu bulunamadı.")
        
    y = df['target']
    
    # Feature Selection (F1_, F2_, F3_, F4_, F5_ prefixleri)
    feature_cols = [col for col in df.columns if col.startswith(('F1_', 'F2_', 'F3_', 'F4_', 'F5_'))]
    
    if not feature_cols:
        raise ValueError("[!] F1_ - F5_ önekiyle başlayan feature bulunamadı. Veri seti soyutlanmamış olabilir.")
        
    X = df[feature_cols]
    
    print("\n" + "="*60)
    print(f"[*] Veri Yüklendi: {X.shape[0]} Örnek, {X.shape[1]} Özellik")
    print(f"[*] Kullanılan Modellerde Feature Grubu Seçimi Dinamik Olarak Tamamlandı:")
    print(f"    - {len(feature_cols)} Adet PSR Uyumlu Feature Tespit Edildi.")
    print("="*60 + "\n")
    return X, y

def optimize_threshold(y_true, y_prob):
    """
    F1-score'u veya PR-AUC'yi dikkate alıp maksimum F1'i veren eşiği bulur.
    """
    best_thresh = 0.5
    best_score = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score

def eval_metrics(y_true, y_prob, threshold=0.5):
    """
    Model için ROC-AUC, PR-AUC, F1, Balanced Acc hesaplar.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
    else:
        roc_auc = 0
        pr_auc = 0
        
    f1 = f1_score(y_true, y_pred, zero_division=0)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    
    return roc_auc, pr_auc, f1, b_acc

def train_and_evaluate(X, y):
    print("\n[*] Stratified 5-Fold Cross Validation & Model Eğitimi Başlatıldı...")
    
    img_lgbm = Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', LGBMClassifier(class_weight='balanced', random_state=42, n_estimators=200, verbosity=-1))])
    img_rf = Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100))])
    img_lr = Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))])
    
    models = {
        'LightGBM': img_lgbm,
        'RandomForest': img_rf,
        'LogisticRegression': img_lr
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {m: {'roc_auc': [], 'pr_auc': [], 'f1': [], 'b_acc': [], 'best_thresh': []} for m in models}
    
    # Array to hold global out-of-fold lightGBM probabilities for analysis
    oof_preds_lgbm = np.zeros(len(X))
    final_calibrated_lgbm = None
    best_lgbm_thresh = 0.5
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        for name, base_model in models.items():
            # CalibratedClassifierCV uses cv=3 internally on train block to calibrate validation probs
            # Using method='sigmoid' (Platt Scaling) or 'isotonic' 
            calibrated = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)
            calibrated.fit(X_train, y_train)
            
            y_prob_val = calibrated.predict_proba(X_val)[:, 1]
            opt_thresh, _ = optimize_threshold(y_val, y_prob_val)
            
            r_auc, p_auc, f1_s, b_act = eval_metrics(y_val, y_prob_val, threshold=opt_thresh)
            
            results[name]['roc_auc'].append(r_auc)
            results[name]['pr_auc'].append(p_auc)
            results[name]['f1'].append(f1_s)
            results[name]['b_acc'].append(b_act)
            results[name]['best_thresh'].append(opt_thresh)
            
            # Save OOF for LightGBM
            if name == 'LightGBM':
                oof_preds_lgbm[val_idx] = y_prob_val
                
    # Rapor
    print("\n" + "-"*50)
    print("CROSS-VALIDATION ORTALAMA PERFORMANS RAPORU (CALIBRATED)")
    print("-" * 50)
    for name in models:
        metrics = results[name]
        print(f"[{name}]")
        print(f"  PR-AUC   : {np.mean(metrics['pr_auc']):.4f} ± {np.std(metrics['pr_auc']):.4f} (Öncelikli Başarı Metriği)")
        print(f"  ROC-AUC  : {np.mean(metrics['roc_auc']):.4f} ± {np.std(metrics['roc_auc']):.4f}")
        print(f"  F1-Score : {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
        print(f"  B-Acc    : {np.mean(metrics['b_acc']):.4f} ± {np.std(metrics['b_acc']):.4f}")
        print(f"  Threshold: {np.mean(metrics['best_thresh']):.2f} (Özel Optimizasyon)")
        
    # Ana Model Eğitimi (LightGBM) Toplam Veri Setinde - Dağıtım İçin
    final_lgbm_base = Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', LGBMClassifier(class_weight='balanced', random_state=42, n_estimators=400, verbosity=-1))])
    final_calibrated_lgbm = CalibratedClassifierCV(estimator=final_lgbm_base, method='sigmoid', cv=3)
    final_calibrated_lgbm.fit(X, y)
    
    # Global Optimal Threshold (OOF üzerinde)
    best_lgbm_thresh, final_f1 = optimize_threshold(y, oof_preds_lgbm)
    
    # Kayıt
    os.makedirs('output', exist_ok=True)
    with open('output/final_lgbm_calibrated.pkl', 'wb') as f:
        pickle.dump(final_calibrated_lgbm, f)
        
    return final_calibrated_lgbm, best_lgbm_thresh, oof_preds_lgbm

def panel_validation(y_true, y_prob_oof, X, threshold):
    print("\n[*] Panel-Benzeri Genelleme Test Simülasyonu (Strateji: Yapısal Feature Gruplaması)...")
    structural_cols = [c for c in X.columns if 'F4_Struct_ConsequenceEncoded' in c]
    if len(structural_cols) > 0:
        group_col = structural_cols[0]
        distinct_vals = X[group_col].unique()
        
        print("  Sınıflandırıcı Genelleme Yeteneği (Varyant Etki Segmentasyonuna Göre):")
        for val in distinct_vals:
            mask = X[group_col] == val
            if mask.sum() > 20: # Sadece anlamlı boyuttaki alt kümeler için rapor ver
                y_sub = y_true[mask]
                prob_sub = y_prob_oof[mask]
                if len(y_sub.unique()) > 1: 
                    roc, pr, f1, b_acc = eval_metrics(y_sub, prob_sub, threshold)
                    print(f"  [Consequence_ID={val:.1f} | N={mask.sum():<5}] PR-AUC: {pr:.4f}, F1: {f1:.4f}, ROC: {roc:.4f}")
                else:
                    pred_sub = (prob_sub >= threshold).astype(int)
                    acc = np.mean(y_sub == pred_sub)
                    print(f"  [Consequence_ID={val:.1f} | N={mask.sum():<5}] Homojen Grup - Doğruluk: {acc:.4f}")

def shap_analysis(calibrated_model, X, output_dir="output"):
    print("\n[*] Explainability Analizi Başlatılıyor (SHAP Gelişmiş Açıklanabilirlik)...")
    
    # SHAP explainer'a Tree (LGBM) objesi verilir.
    # LightGBM NAN'ları kendi destekliyor, imputer adımı pipeline'da RF/LR için olsa da LGBM'in TreeExplainer'ı 
    # numpy array veya DataFrame alabilir.
    
    # 2000 rassal örnek
    sample_X = X.sample(min(2000, len(X)), random_state=42)
    
    # LGBM Pipeline'dan model ağacını al
    base_estimators = calibrated_model.calibrated_classifiers_
    first_pipeline = base_estimators[0].estimator
    lgbm_model = first_pipeline.named_steps['clf']
    imputer = first_pipeline.named_steps['imputer']
    
    # Güvenli imputasyon (sadece array döndürür, pandas index dert etmeyiz)
    sample_X_arr = imputer.transform(sample_X)
    
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(sample_X_arr)
    
    # Gelen shap_values'un şeklini güvenle ayrıştır
    if isinstance(shap_values, list) and len(shap_values) > 1:
        sv = np.array(shap_values[1]) # Pathogenic sınıfı etkisi
    elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = np.array(shap_values)
        
    # Boyut düzeltmeleri
    if len(sv.shape) == 3: # (N, features, classes) in some cases
        sv = sv[:, :, 1]
    elif len(sv.shape) == 1:
        sv = sv.reshape(-1, 1)
        
    # Group By Feature Names
    prefixes = ['F1_', 'F2_', 'F3_', 'F4_', 'F5_']
    group_shap_sums = {p: 0.0 for p in prefixes}
    
    feature_mean_abs = np.abs(sv).mean(axis=0)
    
    # Eşleşme kontrolü
    if feature_mean_abs.shape[0] == sample_X.shape[1]:
        for col_idx, col_name in enumerate(sample_X.columns):
            for p in prefixes:
                if col_name.startswith(p):
                    group_shap_sums[p] += feature_mean_abs[col_idx]
                    break
    else:
        print(f"  [!] SHAP boyut uyuşmazlığı: feature_mean_abs shape {feature_mean_abs.shape}, expected {sample_X.shape[1]}. SHAP toplama atlanıyor.")

    # Summary Plot
    plt.figure()
    try:
        shap.summary_plot(sv, sample_X, show=False)
        plt.savefig(f"{output_dir}/shap_summary.png", bbox_inches='tight')
    except Exception as e:
        print(f"  [!] SHAP Summary Plot çizilemedi: {e}")
    finally:
        plt.close()
    
    print("  Feature Group Seviyesi (Agregate) - SHAP Mutlak Karar Etkisi:")
    for group, val in group_shap_sums.items():
        if val > 0:
            print(f"    - {group}: {val:.4f}")
    
    print(f"  [+] SHAP grafiği kaydedildi: {output_dir}/shap_summary.png")

def error_analysis(y_true, y_prob_oof, X, threshold, output_dir="output"):
    print("\n[*] Error Analysis (Karar Hatası Karakterizasyonu Modülü)...")
    
    y_pred = (y_prob_oof >= threshold).astype(int)
    
    df_err = X.copy()
    df_err['target'] = y_true.values if isinstance(y_true, pd.Series) else y_true
    df_err['prediction'] = y_pred
    df_err['prob'] = y_prob_oof
    
    fp_mask = (df_err['target'] == 0) & (df_err['prediction'] == 1)
    fn_mask = (df_err['target'] == 1) & (df_err['prediction'] == 0)
    
    df_fp = df_err[fp_mask]
    df_fn = df_err[fn_mask]
    
    print(f"  Toplam False Positive Tahmin (Benign -> Pathogenic): {len(df_fp)}")
    print(f"  Toplam False Negative Tahmin (Pathogenic -> Benign): {len(df_fn)}")
    
    if not df_err.empty:
        # False Positive Analizi (Risk Yüksek Çıkmış Ama Benign Olanlar)
        if len(df_fp) > 0:
            in_silico_cols = [c for c in df_err.columns if 'F2_' in c]
            if in_silico_cols:
                # Eşik olarak tüm veri medyanı veya yüksek risk skoru
                high_silico_fp = df_fp[(df_fp[in_silico_cols] > df_err[in_silico_cols].median()).any(axis=1)]
                print(f"  - Analiz 1: Yüksek In-Silico (Risk) skoru alıp Patojenik tahmin edilen ancak Benign olan varyantlar: {len(high_silico_fp)}")
                
        # False Negative Analizi (Pop Frekansı Ortalamadan Düşük Olup Benign Tahmin Edilenler)
        if len(df_fn) > 0:
            pop_cols = [c for c in df_err.columns if 'F1_' in c]
            if pop_cols:
                low_pop_fn = df_fn[(df_fn[pop_cols] < df_err[pop_cols].median()).any(axis=1)]
                print(f"  - Analiz 2: Düşük Popülasyon frekansına (Nadir) sahip olup Benign tahmin edilen Patolojik varyantlar: {len(low_pop_fn)}")
        
        df_fp.to_csv(f"{output_dir}/error_analysis_fp.csv", index=False)
        df_fn.to_csv(f"{output_dir}/error_analysis_fn.csv", index=False)
        print(f"  [+] Hata analiz kümesi (Sorunlu Vakalar) {output_dir}/ dizinine aktarıldı.")

def main():
    parser = argparse.ArgumentParser(description="Nexora AI: ML Eğitim Pipeline'ı (PSR Uyumlu)")
    parser.add_argument("--dataset", default="final_variant_dataset.csv", help="Abstract kolonlu girdisi")
    args = parser.parse_args()
    
    X, y = load_data(args.dataset)
    if X is None:
        return
        
    model, best_thresh, oof_probs = train_and_evaluate(X, y)
    
    print(f"\n[!] Threshold Optimizasyonu: En Yüksek F1 ve PR-AUC Karar Sınırı -> {best_thresh:.3f}")
    
    panel_validation(y, oof_probs, X, best_thresh)
    shap_analysis(model, X, output_dir="output")
    error_analysis(y, oof_probs, X, best_thresh, output_dir="output")
    
    print("\n" + "="*60)
    print("      NEXORA AI - EĞİTİM, UYARLAMA, EXPLAINABILITY TAMAMLANDI")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
