import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_for_ml(df, output_path="final_variant_dataset.csv"):
    """
    Final cleaning for ML readiness (100% PSR compliance).
    - Veri Sızıntısı Önleme: Genomik lokasyon atılır
    - Soyut Kolon İsimlendirme (F1, F2...)
    - Eksik Değer: Median Imputation
    - Sınıf Dengesizliği: Downsampling
    """
    if df.empty:
        print("[!] Veri seti boş, temizleme yapılamadı.")
        return df

    print("[*] Veri seti temizleniyor ve ML için hazırlanıyor (PSR Kriterleri)...")
    
    # 1. Feature Engineering ve Data Leakage Prevention
    # Drop genomic coordinates and identifying string keys directly
    data_leakage_cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'variant_key', 'hgvsp', 'AA_REF', 'AA_ALT']
    df_clean = df.drop(columns=[c for c in data_leakage_cols if c in df.columns], errors='ignore').copy()

    # 2. Kategorik Dönüşümler (Encoding Group 4)
    cat_cols = ['variant_consequence', 'gene_symbol']
    for col in cat_cols:
        if col in df_clean.columns:
            print(f"[*] Kategorik kodlanıyor: {col}")
            le = LabelEncoder()
            df_clean[col] = df_clean[col].astype(str).replace(['nan', 'None', '-'], 'unknown')
            df_clean[col] = le.fit_transform(df_clean[col])

    if 'protein_position' in df_clean.columns:
        df_clean['protein_position'] = pd.to_numeric(
            df_clean['protein_position'].astype(str).str.split('-').str[0], 
            errors='coerce'
        )

    # 3. Soyutlama (Feature Renaming by Groups)
    feature_mapping = {
        'target': 'target',
        'gnomAD_AF': 'F1_Pop_gnomAD_AF',
        'SIFT_score': 'F2_InSilico_SIFT',
        'PolyPhen_score': 'F2_InSilico_PolyPhen',
        'grantham_score': 'F3_Biochem_Grantham',
        'mw_diff': 'F3_Biochem_MW_Diff',
        'polarity_diff': 'F3_Biochem_Polarity_Diff',
        'hydro_diff': 'F3_Biochem_Hydro_Diff',
        'protein_position': 'F4_Struct_Position',
        'variant_consequence': 'F4_Struct_ConsequenceEncoded',
        'gene_symbol': 'F4_Struct_GeneEncoded',
        'PhyloP_score': 'F5_Evo_PhyloP',
        'GERP_score': 'F5_Evo_GERP',
        'PhastCons_score': 'F5_Evo_PhastCons'
    }

    # Rename all columns that exist
    rename_cols = {old: new for old, new in feature_mapping.items() if old in df_clean.columns}
    ml_df = df_clean[list(rename_cols.keys())].copy()
    ml_df.rename(columns=rename_cols, inplace=True)

    # 4. Eksik Verilerin Medyan İmputasyonu (PSR)
    for col in ml_df.columns:
        if col != 'target':
            median_val = ml_df[col].median()
            ml_df[col] = ml_df[col].fillna(median_val)
            
    # 5. Aykırı Değer (Outlier) Analizi ve Temizliği (Capping via 1-99 percentile)
    print("[*] Aykırı değerler analiz ediliyor (Outlier Capping %1 - %99)...")
    numeric_cols = [c for c in ml_df.columns if c != 'target' and not c.endswith('Encoded')]
    for col in numeric_cols:
        lower_bound = ml_df[col].quantile(0.01)
        upper_bound = ml_df[col].quantile(0.99)
        ml_df[col] = np.clip(ml_df[col], lower_bound, upper_bound)
    
    # 6. Sınıf Dengeleme (Downsampling)
    if 'target' in ml_df.columns:
        p_count = (ml_df['target'] == 1).sum()
        b_count = (ml_df['target'] == 0).sum()
        print(f"[*] Sınıf Dağılımı: Pathogenic (1)={p_count}, Benign (0)={b_count}")
        
        # Basit Downsampling stratejisi
        if p_count > 0 and b_count > 0:
            target_size = min(p_count, b_count)
            print(f"[*] Sınıflar {target_size} örneğe dengeleniyor (Downsampling)...")
            df_p = ml_df[ml_df['target'] == 1].sample(target_size, random_state=42)
            df_b = ml_df[ml_df['target'] == 0].sample(target_size, random_state=42)
            ml_df = pd.concat([df_p, df_b]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Çıktıyı başlıklarla birlikte kaydet. Abstract edilmiş olarak.
    ml_df.to_csv(output_path, index=False, header=True)
    print(f"[+] Final veri seti kaydedildi: {output_path} (Şekil: {ml_df.shape})")
    return ml_df

def generate_feature_description(df, output_path="feature_description.txt"):
    """
    PSR kriterlerine %100 uyumlu şekilde Feature Map dosyasını dinamik soyutlamalarla oluşturur.
    """
    description_text = (
        "NEXORA AI - MISSENSE VARYANT SINIFLANDIRMA VERİ SETİ (PSR UYUMLU)\n"
        "=================================================================\n\n"
        "Modellerin yüksek başarımla eğitilebilmesi amacıyla, doğrudan veri sızıntısına (data leakage) yol açabilecek\n"
        "genomik koordinat, referans ve alternatif nükleotid dizilimleri gibi özellikler (chr, pos, ref, alt) veri\n"
        "setinden tamamen arındırılmıştır (Feature Abstraction).\n\n"
        "Bunun yerine PSR beklentilerine uygun olarak özellikler 4 ana gruba ayrılmış ve kolon adları sınırlandırılmıştır:\n\n"
        "[ F1 ] GRUP 1: Popülasyon Özellikleri\n"
        "- F1_Pop_gnomAD_AF: Varyantın gnomAD popülasyonlarındaki genel alel frekansı.\n\n"
        "[ F2 ] GRUP 2: In-Silico Risk Skorları\n"
        "- F2_InSilico_SIFT: Varyantın protein fonksiyonuna potansiyel zararlı etkisini tahmin eden skor.\n"
        "- F2_InSilico_PolyPhen: Fonksiyon hasar olasılığı yapısal skoru.\n\n"
        "[ F3 ] GRUP 3: Biyokimyasal Etkiler\n"
        "- F3_Biochem_Grantham: Amino asit değişiminin biyokimyasal uzaklığı.\n"
        "- F3_Biochem_MW_Diff: Moleküler ağırlık farkı (Moleküler kütle değişimi).\n"
        "- F3_Biochem_Polarity_Diff: Polarite (kutupluluk) farkı.\n"
        "- F3_Biochem_Hydro_Diff: Hidrofobiklik (sudan kaçınma) skor farkı.\n\n"
        "[ F4 ] GRUP 4: Yapısal ve Bağlamsal Özellikler\n"
        "- F4_Struct_Position: Protein dizisindeki asit pozisyonu.\n"
        "- F4_Struct_ConsequenceEncoded: Etki tipinin kodlanmış kategorik karşılığı.\n"
        "- F4_Struct_GeneEncoded: Gen sembolünün kodlanmış kategorik karşılığı.\n\n"
        "[ F5 ] GRUP 5: Evrimsel Korunmuşluk Skorları (Mevcutsa)\n"
        "- F5_Evo_PhyloP: PhyloP evrimsel korunmuşluk skoru.\n"
        "- F5_Evo_GERP: GERP evrimsel korunmuşluk skoru.\n"
        "- F5_Evo_PhastCons: PhastCons evrimsel korunmuşluk skoru.\n\n"
        "* Eksik veriler medyan imputasyon yöntemiyle doldurulmuştur.\n"
        "* Tekrarlayan çapraz veriler (duplicates) elimine edilmiştir.\n"
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(description_text)
    print(f"[+] Özellik açıklamaları (metadata) güncellendi: {output_path}")
