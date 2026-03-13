import pandas as pd
import numpy as np
import os

def generate_mock_data():
    """Generates dummy files to test the pipeline with realistic columns."""
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # 1. ClinVar Mock VCF (Realistic info: No protein in CLNHGVS)
    # 50 Pathogenic, 50 Benign for better training sample
    vcf_rows = []
    # Pathogenic
    for i in range(1, 51):
        pos = 10000 + i
        vcf_rows.append(f"1\t{pos}\trs{i}\tA\tG\t.\tPASS\tALLELEID={i};CLNSIG=Pathogenic;CLNREVSTAT=reviewed_by_expert_panel;MC=SO:0001583|missense_variant;CLNHGVS=NC_000001.11:g.{pos}A>G")
    # Benign
    for i in range(51, 101):
        pos = 20000 + i
        vcf_rows.append(f"1\t{pos}\trs{i}\tC\tT\t.\tPASS\tALLELEID={i};CLNSIG=Benign;CLNREVSTAT=practice_guideline;MC=SO:0001583|missense_variant;CLNHGVS=NC_000001.11:g.{pos}C>T")
    
    clinvar_content = """##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=ALLELEID,Number=1,Type=Integer,Description="the ClinVar Allele ID">
##INFO=<ID=CLNSIG,Number=1,Type=String,Description="Clinical significance">
##INFO=<ID=CLNREVSTAT,Number=1,Type=String,Description="Review status">
##INFO=<ID=MC,Number=.,Type=String,Description="Molecular consequence">
##INFO=<ID=CLNHGVS,Number=.,Type=String,Description="HGVS">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
""" + "\n".join(vcf_rows) + "\n"
    
    with open('data/clinvar.vcf', 'w') as f:
        f.write(clinvar_content)
    print("[+] Mock ClinVar VCF (100 samples) created.")

    # 2. gnomAD Mock VCF (Compressed)
    gnomad_path = 'data/gnomad.exomes.r2.1.1.sites.vcf.bgz'
    gnomad_rows = []
    # Match ClinVar positions
    for i in range(1, 51):
        pos = 10000 + i
        gnomad_rows.append(f"1\t{pos}\t.\tA\tG\t.\tPASS\tAC=1;AN=200000;AF=0.000005")
    for i in range(51, 101):
        pos = 20000 + i
        gnomad_rows.append(f"1\t{pos}\t.\tC\tT\t.\tPASS\tAC=80000;AN=200000;AF=0.4")

    gnomad_content = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + "\n".join(gnomad_rows) + "\n"
    
    import gzip
    with gzip.open(gnomad_path, 'wt', encoding='utf-8') as f:
        f.write(gnomad_content)
    print(f"[+] Mock gnomAD created: {gnomad_path}")

    # 3. dbNSFP Mock TSV (Including AA info)
    dbnsfp_df = pd.DataFrame({
        '#chr': ['1']*100,
        'pos(1-based)': list(range(10001, 10051)) + list(range(20051, 20101)),
        'ref': ['A']*50 + ['C']*50,
        'alt': ['G']*50 + ['T']*50,
        'aaref': ['Ala']*50 + ['Leu']*50,
        'aaalt': ['Val']*50 + ['Ser']*50,
        'SIFT_score': [0.01]*50 + [0.8]*50,
        'Polyphen2_HVAR_score': [0.9]*50 + [0.1]*50,
        'CADD_phred': [30.0]*50 + [5.0]*50,
        'REVEL_score': [0.8]*50 + [0.1]*50,
        'MetaLR_score': [0.9]*50 + [0.05]*50,
        'MetaSVM_score': [0.85]*50 + [0.03]*50,
        'GERP++_RS': [4.0]*50 + [-1.0]*50,
        'phyloP100way_vertebrate': [2.5]*50 + [0.2]*50
    })
    dbnsfp_df.to_csv('data/dbnsfp.tsv', sep='\t', index=False)
    print("[+] Mock dbNSFP created.")

if __name__ == "__main__":
    generate_mock_data()
