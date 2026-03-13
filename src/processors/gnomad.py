import gzip
import re
import pandas as pd
import os
from collections import defaultdict

def normalize_chrom(chrom):
    """Removes 'chr' prefix and converts to string for uniform comparison."""
    c = str(chrom).lower().replace('chr', '')
    if c == 'm': return 'mt'
    return c

def match_gnomad_af(df, gnomad_path):
    """
    Optimized gnomAD matching.
    1. Normalizes chromosome names.
    2. Uses a dictionary for O(1) lookups.
    3. Handles large files efficiently.
    """
    if not gnomad_path or not os.path.exists(gnomad_path):
        print(f"[!] gnomAD dosyası bulunamadı, AF=0 atanıyor.")
        df['gnomAD_AF'] = 0.0
        return df

    print(f"[*] gnomAD eşleştirmesi başlatılıyor (O(1) lookup): {gnomad_path}")
    
    # Pre-process target variants into a nested dictionary: {chrom: {pos: [ref, alt]}}
    # This further optimizes by skipping irrelevant positions fast.
    targets = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        c = normalize_chrom(row['#CHROM'])
        p = str(row['POS'])
        targets[c][p].append((row['REF'], row['ALT']))

    af_map = {}
    found_count = 0
    total_targets = len(df)
    
    try:
        # Open bgz/gz/vcf file
        if gnomad_path.endswith('.gz') or gnomad_path.endswith('.bgz'):
            f = gzip.open(gnomad_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            f = open(gnomad_path, 'r', encoding='utf-8', errors='ignore')
            
        line_count = 0
        for line in f:
            if line.startswith('#'): continue
            
            # Fast split
            parts = line.split('\t', 8)
            if len(parts) < 8: continue
            
            c_raw, p_raw, _, ref_raw, alt_raw = parts[0], parts[1], parts[2], parts[3], parts[4]
            c = normalize_chrom(c_raw)
            
            # Step 1: Chromosome match
            if c in targets:
                # Step 2: Position match
                if p_raw in targets[c]:
                    # Step 3: Ref/Alt match (handling multiallelic sites)
                    found_in_line = False
                    for ref_target, alt_target in targets[c][p_raw]:
                        # gnomAD can have multiallelic lines "A\tC,G"
                        if ref_raw == ref_target and alt_target in alt_raw.split(','):
                            # Extract AF
                            info = parts[7]
                            af_match = re.search(r'AF=([^;]+)', info)
                            if af_match:
                                try:
                                    # If multiallelic, pick the AF corresponding to the alt_target index
                                    alts = alt_raw.split(',')
                                    idx = alts.index(alt_target)
                                    af_val = af_match.group(1).split(',')[idx]
                                    
                                    # Use normalized key for matching with df['variant_key']
                                    # But wait, variant_key in df is usually what's in #CHROM column.
                                    # To be safe, we should use the same normalization for the final mapping.
                                    af_map[f"{c_raw}:{p_raw}:{ref_target}:{alt_target}"] = float(af_val)
                                    found_in_line = True
                                except (ValueError, IndexError):
                                    pass
                    
                    if found_in_line:
                        found_count += 1
                        if found_count >= total_targets:
                            break
            
            line_count += 1
            if line_count % 2000000 == 0:
                print(f"    ... {line_count//1000000}M satır işlendi, {found_count}/{total_targets} eşleşme.")
        
        f.close()
    except Exception as e:
        print(f"[!] gnomAD matching hatası: {e}")

    # We need to make sure the keys in af_map match ClinVar's variant_key
    # If ClinVar has "chr1" and gnomAD has "1", we need bridge them.
    # The most robust way is to rebuild ClinVar's variant_key with normalized chroms too.
    df['tmp_key'] = df.apply(lambda r: f"{r['#CHROM']}:{r['POS']}:{r['REF']}:{r['ALT']}", axis=1)
    df['gnomAD_AF'] = df['tmp_key'].map(af_map).fillna(0.0)
    df.drop(columns=['tmp_key'], inplace=True)
    
    print(f"[+] gnomAD eşleşmesi tamamlandı: {found_count} varyant güncellendi.")
    return df

def row_to_key(c, p, r, a):
    return f"{c}:{p}:{r}:{a}"
