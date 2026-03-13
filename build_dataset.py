"""
Nexora AI - Build Dataset Entry Point
"""
import os
import subprocess
import sys

def main():
    print("[*] Nexora AI Veri Seti Oluşturma Pipeline'ı Başlatılıyor...")
    
    # Path settings
    clinvar_path = "data/clinvar.vcf"
    gnomad_path = "data/gnomad.exomes.r2.1.1.sites.vcf.bgz"
    vep_output = "data/vep_output.txt"
    
    # Run the modular pipeline
    cmd = [
        sys.executable, "-m", "src.pipeline",
        "--clinvar", clinvar_path,
        "--gnomad", gnomad_path,
        "--vep_out", vep_output
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Pipeline hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
