import numpy as np

# Grantham matrix: AA1 (alphabetical) vs AA2
# A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
GRANTHAM_AA = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

_GRANTHAM_MATRIX = [
    [0, 110, 111, 126, 195, 91, 107, 60, 86, 94, 96, 106, 84, 113, 27, 33, 58, 148, 112, 64],
    [110, 0, 86, 96, 180, 43, 54, 125, 29, 97, 102, 26, 91, 97, 103, 110, 71, 101, 77, 96],
    [111, 86, 0, 23, 139, 20, 42, 80, 68, 149, 153, 94, 142, 158, 91, 46, 65, 174, 143, 133],
    [126, 96, 23, 0, 154, 61, 45, 94, 81, 168, 172, 101, 160, 177, 108, 65, 85, 181, 160, 152],
    [195, 180, 139, 154, 0, 154, 170, 159, 188, 198, 198, 202, 196, 205, 169, 155, 149, 215, 194, 192],
    [91, 43, 20, 61, 154, 0, 29, 87, 24, 109, 113, 53, 101, 116, 76, 68, 42, 130, 99, 96],
    [107, 54, 42, 45, 170, 29, 0, 114, 40, 134, 138, 56, 126, 140, 93, 80, 65, 152, 122, 121],
    [60, 125, 80, 94, 159, 87, 114, 0, 98, 135, 138, 127, 127, 153, 42, 56, 59, 184, 147, 109],
    [86, 29, 68, 81, 188, 24, 40, 98, 0, 94, 99, 32, 87, 100, 77, 89, 47, 115, 83, 84],
    [94, 97, 149, 168, 198, 109, 134, 135, 94, 0, 5, 102, 10, 21, 95, 142, 89, 61, 33, 29],
    [96, 102, 153, 172, 198, 113, 138, 138, 99, 5, 0, 107, 15, 22, 98, 145, 92, 61, 36, 32],
    [106, 26, 94, 101, 202, 53, 56, 127, 32, 102, 107, 0, 95, 102, 103, 121, 78, 110, 85, 105],
    [84, 91, 142, 160, 196, 101, 126, 127, 87, 10, 15, 95, 0, 28, 87, 135, 81, 67, 36, 21],
    [113, 97, 158, 177, 205, 116, 140, 153, 100, 21, 22, 102, 28, 0, 114, 155, 103, 40, 22, 50],
    [27, 103, 91, 108, 169, 76, 93, 42, 77, 95, 98, 103, 87, 114, 0, 38, 38, 139, 110, 68],
    [33, 110, 46, 65, 155, 68, 80, 56, 89, 142, 145, 121, 135, 155, 38, 0, 58, 177, 144, 124],
    [58, 71, 65, 85, 149, 42, 65, 59, 47, 89, 92, 78, 81, 103, 38, 58, 0, 134, 92, 69],
    [148, 101, 174, 181, 215, 130, 152, 184, 115, 61, 61, 110, 67, 40, 139, 177, 134, 0, 37, 88],
    [112, 77, 143, 160, 194, 99, 122, 147, 83, 33, 36, 85, 36, 22, 110, 144, 92, 37, 0, 55],
    [64, 96, 133, 152, 192, 96, 121, 109, 84, 29, 32, 105, 21, 50, 68, 124, 69, 88, 55, 0]
]

GRANTHAM_MAP = {aa: i for i, aa in enumerate(GRANTHAM_AA)}

# 3-letter to 1-letter mapping
AA_CODE_MAP = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H',
    'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
    'Tyr': 'Y', 'Val': 'V'
}

# Molecular Weight (g/mol)
AA_MW = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2, 'Q': 146.1, 'E': 147.1, 'G': 75.1, 'H': 155.2,
    'I': 131.2, 'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1, 'S': 105.1, 'T': 119.1, 'W': 204.2,
    'Y': 181.2, 'V': 117.1
}

# Hydrophobicity (Kyte-Doolittle)
AA_HYDRO = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2,
    'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
    'Y': -1.3, 'V': 4.2
}

# Polarity (Zimmerman Scale)
AA_POLARITY = {
    'A': 0.00, 'R': 52.00, 'N': 3.38, 'D': 49.70, 'C': 1.48, 'Q': 3.53, 'E': 49.90, 'G': 0.00, 'H': 51.60,
    'I': 0.13, 'L': 0.13, 'K': 49.50, 'M': 1.43, 'F': 0.35, 'P': 1.58, 'S': 1.67, 'T': 1.66, 'W': 2.10,
    'Y': 1.61, 'V': 0.13
}

def get_grantham_score(aa1, aa2):
    """Calculates Grantham score between two amino acids."""
    try:
        if len(aa1) == 3: aa1 = AA_CODE_MAP.get(aa1, aa1)
        if len(aa2) == 3: aa2 = AA_CODE_MAP.get(aa2, aa2)
        
        aa1 = aa1[0].upper()
        aa2 = aa2[0].upper()
        
        idx1 = GRANTHAM_MAP.get(aa1)
        idx2 = GRANTHAM_MAP.get(aa2)
        
        if idx1 is not None and idx2 is not None:
            return _GRANTHAM_MATRIX[idx1][idx2]
    except Exception:
        pass
    return np.nan

def get_biochem_features(aa_ref, aa_alt):
    """Returns biochemical differences between Ref and Alt amino acids."""
    try:
        if len(aa_ref) == 3: aa_ref = AA_CODE_MAP.get(aa_ref, aa_ref)
        if len(aa_alt) == 3: aa_alt = AA_CODE_MAP.get(aa_alt, aa_alt)
        
        aa_ref = aa_ref[0].upper()
        aa_alt = aa_alt[0].upper()
        
        mw_ref = AA_MW.get(aa_ref, np.nan)
        mw_alt = AA_MW.get(aa_alt, np.nan)
        hydro_ref = AA_HYDRO.get(aa_ref, np.nan)
        hydro_alt = AA_HYDRO.get(aa_alt, np.nan)
        pol_ref = AA_POLARITY.get(aa_ref, np.nan)
        pol_alt = AA_POLARITY.get(aa_alt, np.nan)
        
        return {
            'mw_diff': mw_alt - mw_ref,
            'hydro_diff': hydro_alt - hydro_ref,
            'polarity_diff': pol_alt - pol_ref,
            'abs_mw_diff': abs(mw_alt - mw_ref),
            'abs_hydro_diff': abs(hydro_alt - hydro_ref),
            'abs_polarity_diff': abs(pol_alt - pol_ref)
        }
    except Exception:
        return {
            'mw_diff': np.nan, 'hydro_diff': np.nan, 'polarity_diff': np.nan,
            'abs_mw_diff': np.nan, 'abs_hydro_diff': np.nan, 'abs_polarity_diff': np.nan
        }
