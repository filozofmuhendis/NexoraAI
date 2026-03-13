import pandas as pd
import numpy as np
import os
import re

def filter_clinvar_vcf(vcf_path):
    """
    Parses ClinVar VCF and applies following filters:
    1. Missense variants only (MC=SO:0001583|missense_variant)
    2. Pathogenic / Likely Pathogenic -> 1
    3. Benign / Likely Benign -> 0
    Returns a cleaned DataFrame.
    """
    print(f"[*] ClinVar VCF filtreleniyor: {vcf_path}")
    
    headers = []
    with open(vcf_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('##'): continue
            if line.startswith('#'):
                headers = line.strip().split('\t')
                break
    
    if not headers:
        raise ValueError("VCF başlığı bulunamadı.")

    chunks = pd.read_csv(vcf_path, sep='\t', comment='#', header=None, names=headers, chunksize=50000, low_memory=False)
    
    filtered_dfs = []
    for chunk in chunks:
        info_col = chunk['INFO'].astype(str)
        
        # 1. Missense Check
        # ClinVar uses MC (Molecule Consequence) attribute
        missense_mask = info_col.str.contains('missense_variant', case=False)
        
        # 2. Clinical Significance
        # CLNSIG attribute
        # Pathogenic/Likely Pathogenic
        is_pathogenic = info_col.str.contains('CLNSIG=Pathogenic|CLNSIG=Likely_pathogenic', case=False)
        # Benign/Likely Benign
        is_benign = info_col.str.contains('CLNSIG=Benign|CLNSIG=Likely_benign', case=False)
        
        # Exclude conflicting or uncertain
        is_clean = (is_pathogenic ^ is_benign) & (~info_col.str.contains('Conflicting_interpretations_of_pathogenicity', case=False))
        
        valid_mask = missense_mask & is_clean
        
        chunk = chunk[valid_mask].copy()
        
        if not chunk.empty:
            # Assign labels
            chunk['target'] = np.where(info_col[chunk.index].str.contains('Pathogenic', case=False), 1, 0)
            
            # Variant key for downstream merging
            chunk['variant_key'] = (chunk['#CHROM'].astype(str) + ":" + 
                                   chunk['POS'].astype(str) + ":" + 
                                   chunk['REF'] + ":" + 
                                   chunk['ALT'])
            
            # Extract HGVSp for AA info (optional backup, VEP is primary)
            hgvsp_match = info_col[chunk.index].str.extract(r'CLNHGVS=([^;]+)')
            chunk['hgvsp'] = hgvsp_match[0] if not hgvsp_match.empty else np.nan
            
            filtered_dfs.append(chunk)

    if not filtered_dfs:
        print("[!] Filtrelere uygun varyant bulunamadı.")
        return pd.DataFrame()
        
    df = pd.concat(filtered_dfs).reset_index(drop=True)
    print(f"[+] {len(df)} varyant ClinVar'dan başarıyla filtrelendi.")
    return df
