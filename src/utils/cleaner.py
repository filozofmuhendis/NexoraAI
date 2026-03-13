import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_for_ml(df, output_path="final_variant_dataset.csv"):
    """
    Final cleaning for ML readiness.
    - Encodes categorical features (consequence, gene).
    - Preserves requested features.
    - Handles missing values and balances classes.
    """
    if df.empty:
        print("[!] Veri seti boş, temizleme yapılamadı.")
        return df

    print("[*] Veri seti temizleniyor ve ML için hazırlanıyor...")
    
    # Requested essential features
    # target, gnomAD_AF, SIFT_score, PolyPhen_score, grantham_score, 
    # mw_diff, polarity_diff, protein_position, variant_consequence, gene_symbol
    
    # 1. Handle Categorical Columns (Consequence, Gene)
    cat_cols = ['variant_consequence', 'gene_symbol']
    for col in cat_cols:
        if col in df.columns:
            print(f"[*] Kategori kodlanıyor: {col}")
            le = LabelEncoder()
            # Handle NaN by filling with 'unknown' before encoding
            df[col] = df[col].astype(str).replace(['nan', 'None', '-'], 'unknown')
            df[f'{col}_encoded'] = le.fit_transform(df[col])

    # 2. Handle Numeric Conversion for Protein Position
    if 'protein_position' in df.columns:
        # VEP often gives "123" or "123-124". Take the first number.
        df['protein_position_numeric'] = pd.to_numeric(
            df['protein_position'].astype(str).str.split('-').str[0], 
            errors='coerce'
        )

    # 3. Select Features for ML
    ml_features = [
        'target', 'gnomAD_AF', 'SIFT_score', 'PolyPhen_score', 'grantham_score',
        'mw_diff', 'polarity_diff', 'protein_position_numeric',
        'variant_consequence_encoded', 'gene_symbol_encoded'
    ]
    
    # Include absolute diffs if they exist
    extra = ['abs_mw_diff', 'abs_hydro_diff', 'abs_polarity_diff', 'hydro_diff']
    ml_features.extend([c for c in extra if c in df.columns])

    # Keep only what exists
    available_features = [c for c in ml_features if c in df.columns]
    ml_df = df[available_features].copy()
    
    # 4. Handle Missing Values (Medyan ile doldurma)
    ml_df = ml_df.fillna(ml_df.median())
    
    # 5. Class Balancing (Downsampling)
    if 'target' in ml_df.columns:
        p_count = (ml_df['target'] == 1).sum()
        b_count = (ml_df['target'] == 0).sum()
        print(f"[*] Mevcut dağılım: P={p_count}, B={b_count}")
        
        if p_count > 0 and b_count > 0 and abs(p_count - b_count) > min(p_count, b_count) * 0.1:
            print("[*] Sınıf dengeleme (Downsampling) uygulanıyor...")
            target_size = min(p_count, b_count)
            df_p = ml_df[ml_df['target'] == 1].sample(target_size, random_state=42)
            df_b = ml_df[ml_df['target'] == 0].sample(target_size, random_state=42)
            ml_df = pd.concat([df_p, df_b]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save final dataset
    ml_df.to_csv(output_path, index=False)
    print(f"[+] Final veri seti kaydedildi: {output_path} (Şekil: {ml_df.shape})")
    return ml_df

def generate_feature_description(df, output_path="feature_description.txt"):
    """
    Generates a description file for the final variant dataset.
    """
    descriptions = {
        'target': 'Variant pathogenicity (1: Pathogenic, 0: Benign)',
        'gnomAD_AF': 'Allele Frequency from gnomAD (Population database)',
        'SIFT_score': 'SIFT score from VEP (Lower is more damaging)',
        'PolyPhen_score': 'PolyPhen-2 score from VEP (Higher is more damaging)',
        'grantham_score': 'Grantham distance between original and mutant AA',
        'mw_diff': 'Molecular weight difference (Alt - Ref)',
        'polarity_diff': 'Polarity difference (Alt - Ref)',
        'protein_position_numeric': 'Numerical position of the variant in the protein',
        'variant_consequence_encoded': 'Label encoded variant consequence (e.g. missense_variant)',
        'gene_symbol_encoded': 'Label encoded gene symbol',
        'abs_mw_diff': 'Absolute molecular weight difference',
        'abs_polarity_diff': 'Absolute polarity difference',
        'hydro_diff': 'Hydrophobicity difference',
        'abs_hydro_diff': 'Absolute hydrophobicity difference'
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Nexora AI - Feature Descriptions\n")
        f.write("=================================\n\n")
        for col in df.columns:
            desc = descriptions.get(col, "Engineered feature for machine learning.")
            f.write(f"{col}: {desc}\n")
    
    print(f"[+] Özellik açıklamaları güncellendi: {output_path}")
