from pathlib import Path

import wandb
import pandas as pd


def get_last_checkpoint(path):
    p = Path(path) / 'checkpoints/last-v1.ckpt'
    if p.exists():
        return p
    return Path(path) / 'checkpoints/last.ckpt'


DF_VARIANTS_PATH = 'data/ukb_variants_pvar.tsv'
GWAS_VARIANTS_PATH = 'data/gwas_prioritization/variants_imp.tsv'
GWAS_VARIANTS_PATH_CHR = {
    chrom: f'data/gwas_prioritization/variants_imp_chr_{chrom}.tsv'
    for chrom in range(1, 23)
}
PLINK_PATH = './plink'
GWAS_DATA_DIR = 'data/gwas_prioritization/gen_data'

ENH_ATLAS_N_PERM_TESTS = 100
N_PERMS_TESTS = 10000

MODISCO_N_VARIANTS = 5_000_000
MODISCO_N_SEQLETS = 500_000
MODISCO_MOTIF_DB = 'data/motif_databases/HUMAN/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme'

LD_REF = '../prspipe/resources/1kg/1KGPhase3.w_hm3.GW'
P_VAL_RANGES = [
    1
]
MIN_VARIANTS_COVERAGES = [
   0,
]
PVAL_TYPES = [
    'alt',
]
DATASETS = ('ukbfull',)
PHENOTYPES_PATH = 'data/phenotypes_endpoints_bb.csv'
CAUSAL_DB_PATH = 'data/CAUSALdb/credible_set.txt'
NEG_DB_PATH = 'data/CAUSALdb/neg_set.txt'
FINEMAPPING_TOOLS = [
    'abf',
    'susie',
    'finemap',
    'paintor',
    'caviarbf',
    'polyfun_finemap',
    'polyfun_susie',
]

FINEMAPPING_TOOLS = FINEMAPPING_TOOLS + [
    f'{method}_top{topk}_cws' for method in FINEMAPPING_TOOLS for topk in [
        5,
        10,
    ]
]


model_paths = {
    # specify MIFM model ID and checkpoint path here, e.g.,:
    # 'abc123': get_last_checkpoint('logs/deep-vep/abc123/'),
}


df_meta = pd.read_csv('data/CAUSALdb/meta.txt', sep='\t')
meta_ids_field_ids = [
    # e.g.:
    # ('CA243', 'C3_BREAST'),
]
df_meta_field = pd.DataFrame(meta_ids_field_ids, columns=['meta_id', 'FieldID'])
df_meta = df_meta_field.merge(df_meta, on='meta_id', how='left')
