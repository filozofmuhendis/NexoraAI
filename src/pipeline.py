import os
import sys
import argparse
import pandas as pd
import numpy as np

# Pipeline modules
from src.processors.clinvar import filter_clinvar_vcf
from src.processors.vep import generate_vep_input, run_vep, parse_vep_output, check_vep_installed
from src.processors.gnomad import match_gnomad_af
from src.features.biochem import get_grantham_score, get_biochem_features
from src.utils.cleaner import clean_for_ml, generate_feature_description

def main():
    parser = argparse.ArgumentParser(description="Nexora AI: Otomatik Varyant Veri Hazırlama Pipeline'ı")
    parser.add_argument("--clinvar", default="data/clinvar.vcf", help="ClinVar VCF dosyası yolu")
    parser.add_argument("--gnomad", default="data/gnomad.exomes.r2.1.1.sites.vcf.bgz", help="gnomAD VCF yolu")
    parser.add_argument("--vep_out", default="data/vep_output.txt", help="VEP çıktı dosyası (varsa okunur, yoksa oluşturulur)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("      NEXORA AI - GELİŞMİŞ VARYANT PİPELİNE BAŞLATILDI       ")
    print("="*60 + "\n")

    # AŞAMA 1: ClinVar Filtreleme
    if not os.path.exists(args.clinvar):
        print(f"[!] Hata: ClinVar dosyası bulunamadı: {args.clinvar}")
        sys.exit(1)
    
    df = filter_clinvar_vcf(args.clinvar)
    if df.empty:
        print("[!] İşleme devam edilemiyor, varyant bulunamadı.")
        return

    # AŞAMA 2: VEP Anotasyonu (Real Run)
    # Önce giriş dosyasını hazırlıyoruz.
    vep_input = generate_vep_input(df)
    
    # VEP'i çalıştır (Eğer yüklü değilse talimat verir)
    if check_vep_installed():
        vep_file = run_vep(vep_input, args.vep_out)
    else:
        # VEP yüklü değilse ve çıktı dosyası da yoksa uyarı ver
        if not os.path.exists(args.vep_out):
            run_vep(vep_input, args.vep_out) # Bu adım talimatları yazdıracaktır
            vep_file = None
        else:
            print(f"[*] VEP yüklü değil ancak mevcut çıktı dosyası bulundu: {args.vep_out}")
            vep_file = args.vep_out

    # VEP çıktılarını oku ve birleştir
    vep_df = parse_vep_output(vep_file)
    if not vep_df.empty:
        print(f"[*] VEP verileri ({len(vep_df)} kayıt) birleştiriliyor...")
        df = df.merge(vep_df, on='variant_key', how='left')
        
        # Merge başarısı kontrolü
        valid_sift = df['SIFT_score'].notna().sum()
        print(f"[*] Birleştirme sonrası SIFT_score olan varyant sayısı: {valid_sift} / {len(df)}")
    else:
        print("[!] Uyarı: VEP anotasyon verileri sağlanamadı. Skorlar eksik kalacak.")

    # AŞAMA 3: gnomAD Popülasyon Frekansı (O(1) Optimized)
    if os.path.exists(args.gnomad):
        df = match_gnomad_af(df, args.gnomad)
    else:
        print(f"[!] Uyarı: gnomAD dosyası bulunamadı, AF değerleri 0 atanıyor.")
        df['gnomAD_AF'] = 0.0

    # AŞAMA 4: Biyokimyasal Özellik Üretimi
    print("[*] Biyokimyasal özellikler hesaplanıyor (Grantham, MW, Polarity)...")
    
    def compute_features(row):
        # AA_REF ve AA_ALT öncelikli olarak VEP'ten (veya ClinVar fallback) gelmeli
        ref = row.get('AA_REF')
        alt = row.get('AA_ALT')
        
        if pd.isna(ref) or pd.isna(alt):
            return pd.Series({
                'grantham_score': np.nan, 'mw_diff': np.nan, 
                'hydro_diff': np.nan, 'polarity_diff': np.nan
            })
        
        g_score = get_grantham_score(ref, alt)
        biochem = get_biochem_features(ref, alt)
        
        res = {'grantham_score': g_score}
        res.update(biochem)
        return pd.Series(res)

    feat_df = df.apply(compute_features, axis=1)
    df = pd.concat([df, feat_df], axis=1)

    # AŞAMA 5: Temizleme ve ML Hazırlığı (Encoding & Balancing)
    final_df = clean_for_ml(df, "final_variant_dataset.csv")
    generate_feature_description(final_df, "feature_description.txt")

    print("\n" + "="*60)
    print(f" [SUCCESS] Pipeline başarıyla tamamlandı.")
    print(f" [*] Final Veri Seti: final_variant_dataset.csv ({len(final_df)} varyant)")
    print(f" [*] Özellik Sayısı: {len(final_df.columns)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
