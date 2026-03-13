# Nexora AI: Genetik Varyant Sınıflandırma Pipeline'ı

Bu proje, genetik varyantların (missense) patojenik veya benign olarak sınıflandırılması için yarışma düzeyinde bir veri seti oluşturma ve modelleme altyapısı sunar.

## Proje Yapısı
- `src/pipeline.py`: Veri işleme, filtreleme ve birleştirme (ClinVar, gnomAD, dbNSFP).
- `src/features.py`: Biyokimyasal öznitelik hesaplamaları (Grantham, MW, Hidrofobiklik).
- `src/model.py`: XGBoost baseline model eğitimi ve 5-fold cross-validation.
- `build_dataset.py`: Tüm veri pipeline'ını çalıştıran ana betik.
- `train_baseline.py`: Veri seti oluştuktan sonra modeli eğiten betik.
- `data_gen_mock.py`: Test amaçlı sentetik veri üretici.

## Gereksinimler
- Python 3.10+
- `pip install -r requirements.txt` (pandas, numpy, scikit-learn, xgboost)

## Kullanım

### 1. Veri Hazırlama
`data/` klasörüne aşağıdaki dosyaları yerleştirin:
- `clinvar.vcf`: ClinVar VCF dosyası.
- `gnomad.tsv`: gnomAD frekans verileri.
- `dbnsfp.tsv`: dbNSFP skorları.

*Not: Eğer bu dosyalar yoksa, `build_dataset.py` otomatik olarak küçük bir sentetik (mock) veri seti oluşturacaktır.*

### 2. Pipeline'ı Çalıştırma
Veri setini oluşturmak için:
```bash
python build_dataset.py
```
Bu işlem sonunda `final_variant_dataset.csv` ve `feature_description.txt` dosyaları oluşacaktır.

### 3. Model Eğitimi
Oluşturulan veri seti ile baseline model eğitmek için:
```bash
python train_baseline.py
```

## Veri Pipeline Özellikleri
- **ClinVar Filtreleme**: Yalnızca 3 ve 4 yıldız review status'a sahip missense varyantlar.
- **Enrichment**: gnomAD'dan AF, dbNSFP'den SIFT, PolyPhen, CADD, REVEL, MetaLR, MetaSVM, GERP++, PhyloP skorları.
- **Feature Engineering**: Grantham skoru ve amino asitler arası ΔMW, ΔHidrofobiklik hesaplamaları.
- **Sızıntı Önleme (Anti-Leakage)**: Koordinat ve spesifik nükleotid bilgileri model eğitiminden önce temizlenir.
- **Dengeleme**: Sınıf dağılımı dengesizse otomatik downsampling uygulanır.

## Model Performansı
Model, 5-fold stratified cross-validation ile değerlendirilir ve ROC-AUC, F1-score gibi metrikler raporlanır.
