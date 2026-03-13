import pandas as pd
import numpy as np
import subprocess
import os
import re
import shutil

# Common manual install paths for this user
MANUAL_VEP_PATH = r"C:\Users\pc\ensembl-vep\vep"
GIT_PERL_PATH = r"C:\Program Files\Git\usr\bin\perl.exe"

def get_vep_command():
    """
    Returns the command to run VEP.
    Prioritizes Docker on Windows for stability, then system path, then manual install.
    """
    # 1. Check Docker (Best for Windows/WSL)
    if shutil.which("docker"):
        # We use a placeholder for the volume mapping which will be finalized in run_vep
        return ["docker", "run", "--rm", "ensemblorg/ensembl-vep", "vep"]
    
    # 2. Check system PATH
    system_vep = shutil.which("vep")
    if system_vep:
        return [system_vep]
    
    # 3. Check manual path
    if os.path.exists(MANUAL_VEP_PATH):
        perl_cmd = shutil.which("perl") or (GIT_PERL_PATH if os.path.exists(GIT_PERL_PATH) else "perl")
        return [perl_cmd, MANUAL_VEP_PATH]
    
    return None

def check_vep_installed():
    """Checks if VEP is available via any method."""
    return get_vep_command() is not None

def run_vep(input_vcf, output_file="data/vep_output.txt"):
    """
    Executes Ensembl VEP.
    Handles both local and Docker execution logic.
    """
    cmd_base = get_vep_command()
    
    if not cmd_base:
        print("\n" + "!"*60)
        print("[!] Hata: Ensembl VEP bulunamadı (Docker, PATH veya Manuel).")
        print("[!] Lütfen Docker Desktop'ı başlatın veya VEP'i kurun.")
        print("!"*60 + "\n")
        return None

    is_docker = "docker" in cmd_base
    print(f"[*] VEP çalıştırılıyor (Yöntem: {'Docker' if is_docker else 'Lokal Persistans'})...")
    
    # Finalize command with arguments
    if is_docker:
        # Docker needs volume mapping to access the local project directory
        cwd = os.getcwd()
        # In Docker, we map the project root to /data. 
        # The cache is already inside the project (vep_cache/).
        # We tell VEP to look for the cache in /data/vep_cache
        
        full_cmd = [
            "docker", "run", "--rm",
            "-v", f"{cwd}:/data",
            "ensemblorg/ensembl-vep",
            "vep",
            "-i", f"/data/{input_vcf}",
            "-o", f"/data/{output_file}",
            "--offline", "--assembly", "GRCh37",
            "--dir_cache", "/data/vep_cache",
            "--format", "vcf", "--force_overwrite", "--no_stats",
            "--sift", "b", "--polyphen", "b", "--symbol", "--canonical", "--tab"
        ]
        # In Docker, cache is usually not shared unless we map it. 
        # If the user doesn't have a cache, this might fail unless they use --database
        # For now, we try offline as requested, but if it fails, we warn about cache.
    else:
        full_cmd = cmd_base + [
            "-i", input_vcf, "-o", output_file,
            "--cache", "--offline", "--assembly", "GRCh37",
            "--format", "vcf", "--force_overwrite", "--no_stats",
            "--sift", "b", "--polyphen", "b", "--symbol", "--canonical", "--tab"
        ]

    # Environment and CWD for local execution
    env = os.environ.copy()
    exec_cwd = None
    if not is_docker and os.path.exists(MANUAL_VEP_PATH):
        vep_dir = os.path.dirname(MANUAL_VEP_PATH)
        modules_dir = os.path.join(vep_dir, 'modules')
        if os.path.exists(modules_dir):
            env['PERL5LIB'] = modules_dir + os.pathsep + env.get('PERL5LIB', '')
        exec_cwd = vep_dir

    try:
        subprocess.run(full_cmd, check=True, env=env, cwd=exec_cwd)
        print(f"[+] VEP anotasyonu tamamlandı: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"[!] VEP hatası: {e}")
        if is_docker:
            print("[*] İpucu: Docker üzerinde VEP çalışırken 'cache' bulunamamış olabilir.")
            print("[*] Cache indirmek için: docker run -v $PWD/vep_cache:/opt/vep/.vep ensemblorg/ensembl-vep INSTALL.pl")
        else:
            print("[*] İpucu: Lokal Perl bağımlılıkları eksik. Docker Desktop'ı başlatıp tekrar deneyin.")
        return None

def parse_vep_output(vep_path):
    """
    Parses VEP tab-delimited output.
    """
    if not vep_path or not os.path.exists(vep_path):
        return pd.DataFrame()

    print(f"[*] VEP sonuçları okunuyor: {vep_path}")
    
    try:
        header = None
        with open(vep_path, 'r') as f:
            for line in f:
                if line.startswith('#Uploaded_variation'):
                    header = line.strip('#').strip().split('\t')
                    break
        
        if not header:
            return pd.DataFrame()

        df_vep = pd.read_csv(vep_path, sep='\t', comment='#', header=None, names=header)
        
        if 'CANONICAL' in df_vep.columns:
            df_vep = df_vep[df_vep['CANONICAL'] == 'YES'].copy()

        def get_score(val):
            if pd.isna(val) or val == '-': return np.nan
            m = re.search(r'\(([\d.]+)\)', str(val))
            return float(m.group(1)) if m else np.nan

        if 'SIFT' in df_vep.columns:
            df_vep['SIFT_score'] = df_vep['SIFT'].apply(get_score)
        if 'PolyPhen' in df_vep.columns:
            df_vep['PolyPhen_score'] = df_vep['PolyPhen'].apply(get_score)

        if 'Amino_acids' in df_vep.columns:
            df_vep[['AA_REF', 'AA_ALT']] = df_vep['Amino_acids'].str.split('/', expand=True).iloc[:, :2]

        rename_map = {
            'Uploaded_variation': 'variant_key_vep',
            'SYMBOL': 'gene_symbol',
            'Consequence': 'variant_consequence',
            'Protein_position': 'protein_position'
        }
        df_vep = df_vep.rename(columns=rename_map)
        df_vep['variant_key'] = df_vep['variant_key_vep'].str.replace('_', ':')
        
        cols = ['variant_key', 'SIFT_score', 'PolyPhen_score', 'AA_REF', 'AA_ALT', 
                'protein_position', 'variant_consequence', 'gene_symbol']
        return df_vep[[c for c in cols if c in df_vep.columns]].drop_duplicates('variant_key')

    except Exception as e:
        print(f"[!] VEP ayrıştırma hatası: {e}")
        return pd.DataFrame()

def generate_vep_input(df, output_path="data/vep_input.vcf"):
    """Saves variants for VEP processing."""
    if df.empty: return None
    print(f"[*] VEP giriş dosyası hazırlanıyor: {output_path}")
    vcf_cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
    for col in vcf_cols:
        if col not in df.columns:
            df[col] = '.' if col != 'FILTER' else 'PASS'
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[vcf_cols].to_csv(output_path, sep='\t', index=False)
    return output_path
