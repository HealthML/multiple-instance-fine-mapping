import h5py
import subprocess
import urllib.request
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyplink import PyPlink
import statsmodels.api as sm
from contextlib import redirect_stdout, redirect_stderr, contextmanager

from snakemake import shell

import src.snakemake_config as config
from src import predict_chr, pgs, clinvar, motifs, enhancer_atlas, mpra_saturation, run_lmm, \
    predict_gwas_prioritization_chr

results_root = 'results_debug/' if config.debug else 'results/'

DL_MODEL_IDS = set(config.model_paths.keys())
DL_MODEL_IDS_SECONDARY = set(config.model_paths_secondary.keys())
ALL_DL_MODEL_IDS = set(list(DL_MODEL_IDS) + list(DL_MODEL_IDS_SECONDARY))
MODEL_PATHS = config.model_paths.copy()
MODEL_PATHS.update(config.model_paths_secondary)
ALL_MODEL_IDS = ['all'] + list(DL_MODEL_IDS)


@contextmanager
def redirect_output(log):
    try:
        with open(str(log['stdout']),'wt',buffering=1) as f_stdout:
            with open(str(log['stderr']),'wt',buffering=1) as f_stderr:
                with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                    yield
    finally:
        pass


def get_trait_for_meta_id(meta_id):
    return config.df_meta.loc[config.df_meta['meta_id'] == meta_id].iloc[0]['trait']


rule liftover_neg_samples:
    output:
        'data/CAUSALdb/neg_set_hg38.txt'
    log:
        stdout='data/CAUSALdb/data/CAUSALdb/CAUSALdb/neg_set_hg38.log',
        stderr='data/CAUSALdb/data/CAUSALdb/CAUSALdb/neg_set_hg38.stderr'
    resources:
        partition='gpu-batch',
        time='3-00:00:00',
        mem='128gb',
        threads=2,
    run:
        with redirect_output(log):
            df_hg38 = pd.read_csv(
                'hglft_genome_2de94_56f750.bed',
                sep='\t',
                header=None,
                names=['chr', 'start', 'end'],
                index_col=False,
            )
            df_hg19 = pd.read_csv(
                'liftover_input_neg.bed',
                sep=' ',
                header=None,
                names=['chr', 'start', 'end'],
                index_col=False,
            )
            df_skip = pd.read_csv(
                'hglft_genome_2de94_56f750.err.txt',
                header=None,
                sep='\t',
                skiprows=lambda i: i % 2 == 0,
                names=['chr', 'start', 'end'],
                index_col=False,
            )

            df = pd.read_csv(
                'data/CAUSALdb/neg_set.txt',
                sep='\t',
            )
            from collections import defaultdict

            i_skipped = 0

            hg19_to_h38 = defaultdict(dict)

            max_rows = None
            for i, row in tqdm(df_hg19.iloc[:max_rows].iterrows(),total=len(df_hg19),miniters=1000):
                if len(df_skip.loc[
                    (df_skip['chr'] == row['chr'])
                    & (df_skip['end'] == row['end'])
                ]) < 1:
                    chrom, bp = df_hg38.iloc[i - i_skipped][['chr', 'end']]

                    try:
                        chrom = int(chrom.replace('chr',''))
                        hg19_to_h38[chrom][row['end']] = chrom, bp
                    except ValueError:
                        pass
                else:
                    i_skipped += 1

            import pickle

            with open('hg19_to_h38_neg.pickle','wb') as handle:
                pickle.dump(hg19_to_h38,handle,protocol=pickle.HIGHEST_PROTOCOL)

            groups = []
            for chr_bp, group in tqdm(df.groupby(['chr', 'bp'])):
                try:
                    group[['chr_hg38', 'bp_hg38']] = hg19_to_h38[chr_bp[0]][chr_bp[1]]
                except KeyError:
                    pass
                groups.append(group)
            df = pd.concat(groups)
            df['bp_hg38'] = df['bp_hg38'].astype('Int64')
            df['chr_hg38'] = df['chr_hg38'].astype('Int64')

            df.to_csv(
                'data/CAUSALdb/neg_set_hg38.txt',
                index=False,
                sep='\t'
            )


rule match_genostan_track:
    output:
        'data/CAUSALdb/genostan/{track_id}.txt'
    log:
        stdout='data/CAUSALdb/genostan/{track_id}.log',
        stderr='data/CAUSALdb/genostan/{track_id}.stderr'
    resources:
        partition='gpu-batch,cpu-batch',
        time='3-00:00:00',
        mem='64gb',
        threads=2,
    run:
        with redirect_output(log):
            st_path = f"data/genostan/E{wildcards['track_id']}_GenoSTAN_nb_20_states.bed.gz"
            df_st = pd.read_csv(
                st_path,
                sep='\t',
                header=None,
                names=['chr', 'start', 'end', 'state'],
                index_col=False,
            )
            track_name = Path(st_path).name.split('_')[0]

            df = pd.read_csv(
                'data/CAUSALdb/credible_set_hg38.txt',
                sep='\t',
                usecols=['chr_hg38', 'bp_hg38']
            ).drop_duplicates()
            df['chr_hg38'] = df['chr_hg38'].astype(str).str.replace('.0','')

            dfs = []
            for chrom, group in df_st.groupby('chr'):
                chrom = chrom.replace('chr','')
                df_chr = df.loc[df['chr_hg38'] == chrom]

                if len(df_chr) < 1:
                    continue

                for _, row in tqdm(group.iterrows(),total=len(group)):
                    df_chr.loc[
                        (df_chr['bp_hg38'].between(row['start'],row['end'])),
                        f'state_{track_name}'] = row['state']
                dfs.append(df_chr)

            pd.concat(dfs).to_csv(
                str(output),
                sep='\t',
                index=False,
            )


rule match_genostan_track_neg:
    output:
        'data/CAUSALdb/genostan_neg/{track_id}.txt'
    log:
        stdout='data/CAUSALdb/genostan_neg/{track_id}.log',
        stderr='data/CAUSALdb/genostan_neg/{track_id}.stderr'
    resources:
        partition='gpu-batch,cpu-batch',
        time='3-00:00:00',
        mem='64gb',
        threads=2,
    run:
        with redirect_output(log):
            st_path = f"data/genostan/E{wildcards['track_id']}_GenoSTAN_nb_20_states.bed.gz"
            df_st = pd.read_csv(
                st_path,
                sep='\t',
                header=None,
                names=['chr', 'start', 'end', 'state'],
                index_col=False,
            )
            track_name = Path(st_path).name.split('_')[0]

            df = pd.read_csv(
                'data/CAUSALdb/neg_set_hg38.txt',
                sep='\t',
                usecols=['chr_hg38', 'bp_hg38']
            ).drop_duplicates()
            df['chr_hg38'] = df['chr_hg38'].astype(str).str.replace('.0','')

            dfs = []
            for chrom, group in df_st.groupby('chr'):
                chrom = chrom.replace('chr','')
                df_chr = df.loc[df['chr_hg38'] == chrom]

                if len(df_chr) < 1:
                    continue

                for _, row in tqdm(group.iterrows(),total=len(group)):
                    df_chr.loc[
                        (df_chr['bp_hg38'].between(row['start'],row['end'])),
                        f'state_{track_name}'] = row['state']
                dfs.append(df_chr)

            pd.concat(dfs).to_csv(
                str(output),
                sep='\t',
                index=False,
            )


GENOSTAN_TRACK_IDS = []
for i in range(1,130):
    i_str = str(i)
    if i in (60, 64):
        continue
    while len(i_str) < 3:
        i_str = '0' + i_str
    GENOSTAN_TRACK_IDS.append(i_str)

rule all_match_genostan_track:
    input:
        labels_pos=[f'data/CAUSALdb/genostan/{track_id}.txt' for track_id in GENOSTAN_TRACK_IDS],
        labels_neg=[f'data/CAUSALdb/genostan_neg/{track_id}.txt' for track_id in GENOSTAN_TRACK_IDS]
    output:
        output_pos='data/CAUSALdb/credible_set_genostan.txt',
        output_neg='data/CAUSALdb/neg_set_genostan.txt'
    log:
        stdout='data/CAUSALdb/credible_set_genostan.log',
        stderr='data/CAUSALdb/credible_set_genostan.stderr'
    resources:
        partition='gpu-batch,cpu-batch',
        time='3-00:00:00',
        mem='128gb',
        threads=2,
    run:
        with redirect_output(log):
            from functools import reduce

            labels_values = sorted([
                'Elon.7',
                'ElonW.13',
                'ElonW.15',
                'Enh.6',
                'EnhW.8',
                'EnhWF.9',
                "Gen5'.16",
                'Low.12',
                'Low.17',
                'Low.18',
                'Low.20',
                'Low.3',
                'Low.4',
                'Low.5',
                'Prom.1',
                'Prom.19',
                'Repr.14',
                'Repr.2',
                'ReprEnh.11',
                'ReprW.10'
            ])

            for labels, output_file, orig_data in zip(
                    (
                            input['labels_pos'],
                            input['labels_neg']
                    ),
                    (
                            output['output_pos'],
                            output['output_neg']
                    ),
                    (
                            'data/CAUSALdb/credible_set_hg38.txt',
                            'data/CAUSALdb/neg_set_hg38.txt',
                    )
            ):
                df_genostan = reduce(
                    lambda left, right: pd.merge(
                        left,
                        right,
                        on=['chr_hg38', 'bp_hg38'],
                        how='left',
                    ),
                    [
                        pd.read_csv(
                            p,
                            sep='\t',
                        ) for p in labels
                    ],
                )


                def genostan_state_to_int(state):
                    try:
                        return labels_values.index(state) + 1
                    except ValueError:
                        return 0


                for c in df_genostan.columns:
                    if c not in ['chr_hg38', 'bp_hg38']:
                        df_genostan[c] = df_genostan[c].apply(genostan_state_to_int)
                        df_genostan[c] = df_genostan[c].astype(int)

                df = pd.read_csv(
                    orig_data,
                    sep='\t',
                )

                df.merge(
                    df_genostan,
                    on=['chr_hg38', 'bp_hg38'],
                    how='left',
                ).to_csv(
                    output_file,
                    sep='\t',
                    index=False,
                )

rule add_genostan_to_mpra:
    output:
        'data/kircher_mpra/GRCh37_ALL_genostan.tsv'
    log:
        stdout='data/kircher_mpra/GRCh37_ALL_genostan.log',
        stderr='data/kircher_mpra/GRCh37_ALL_genostan.stderr'
    resources:
        partition='cpu-batch,gpu-batch',
        time='3-00:00:00',
        gpus=0,
        mem='128gb',
        threads=2,
    run:
        with redirect_output(log):
            df_38 = pd.read_csv(
                'data/kircher_mpra/GRCh38_ALL.tsv',
                sep='\t',
            )
            df_38 = df_38.drop_duplicates(['Chromosome', 'Position'])
            df_38['Position'] = df_38['Position'].astype(int)

            df_37 = pd.read_csv(
                'data/kircher_mpra/GRCh37_ALL.tsv',
                sep='\t',
            )
            df_37['Position'] = df_37['Position'].astype(int)
            for idx, row in df_38.iterrows():
                assert row["Element"] == df_37.at[idx, 'Element']

            df_sts = []
            for p in tqdm(Path('data/genostan/').glob('*.bed.gz'),total=130):
                df_st = pd.read_csv(
                    p,
                    sep='\t',
                    header=None,
                    names=['chr', 'start', 'end', 'state'],
                    index_col=False,
                )
                df_st['chr'] = df_st['chr'].apply(lambda c: c.replace('chr',''))
                track_name = p.name.split('_')[0]
                df_st['track_name'] = track_name
                df_sts.append(df_st)

            df_st = pd.concat(df_sts)

            for idx, row in tqdm(df_38.iterrows(),total=len(df_38),miniters=50):
                df_geno = df_st.loc[
                    (df_st['chr'] == row['Chromosome'])
                    & (df_st['start'] <= row['Position'])
                    & (df_st['end'] >= row['Position'])
                    ]
                if len(df_geno) > 0:
                    for _, row_st in df_geno.iterrows():
                        df_37.at[idx, f"state_{row_st['track_name']}"] = row_st['state']

            df_37.to_csv(
                'data/kircher_mpra/GRCh37_ALL_genostan.tsv',
                sep='\t',
                index=False,
            )

rule predict_causaldb_variants:
    output:
        results_root + "model_predictions/{model_id}/model_predictions_chr{chrom}.hdf5"
    log:
        stdout=results_root + "model_predictions/{model_id}/model_predictions_chr{chrom}.log",
        stderr=results_root + "model_predictions/{model_id}/model_predictions_chr{chrom}.stderr"
    resources:
        # partition='cpu-batch,gpu-batch',
        partition='cpu-batch',
        time='3-00:00:00',
        #partition='gpu-batch',
        #gpus='v100:1',
	#gpus=1,
	#constraint='GPU_SKU:A100',
        mem='256gb',
        # partition='cpu-batch',
        # mem='128gb',
        threads=8,
    priority: 1
    run:
        with redirect_output(log):
            if (
                    wildcards.model_id not in (
                        'all',
                        'enformer',
                        'enformer_normalized',
                        'cadd',
                    ) and wildcards.model_id not in config.FINEMAPPING_TOOLS
            ):
                predict_chr.generate_predictions(
                    model_checkpoint=MODEL_PATHS[wildcards.model_id],
                    variants_path=config.CAUSAL_DB_PATH,
                    results_dir=results_root + '/model_predictions/',
                    load_pretrained_enformer=wildcards.model_id == 'enformer',
                    load_pretrained_basenji='basenji2' in wildcards.model_id,
                    basenji_normalized='basenji2_normalized' in wildcards.model_id,
                    predict_all_chrom=wildcards.chrom,
                )
            else:
                # create a dummy file
                for chr_output in output:
                    open(str(chr_output),'a').close()


rule all_get_basenji2_causaldb_scores_for_chromosome:
    input:
        [
            results_root + "model_predictions/basenji2/model_predictions_chr" + str(chrom) + ".hdf5"
            for chrom in range(1,23)
         ] + [
            results_root + "model_predictions/basenji2_normalized/model_predictions_chr" + str(chrom) + ".hdf5"
            for chrom in range(1,23)
        ]


rule get_enformer_1000_genomes_scores_for_chromosome:
    output:
        results_root + "model_predictions_enformer_1000_genomes/model_predictions_chr{chrom}.hdf5"
    log:
        stdout=results_root + "model_predictions_enformer_1000_genomes/model_predictions_chr{chrom}.log",
        stderr=results_root + "model_predictions_enformer_1000_genomes/model_predictions_chr{chrom}.stderr"
    resources:
        partition='cpu-batch',
        time='3-00:00:00',
        mem='16gb',
        threads=2,
    run:
        with redirect_output(log):
            preds_file = Path(str(output)).parent / f'SAD_chr_{wildcards.chrom}.hdf5'
            urllib.request.urlretrieve(
                f'https://storage.googleapis.com/dm-enformer/variant-scores/1000-genomes/enformer/1000G.MAF_threshold%3D0.005.{wildcards.chrom}.h5',
                preds_file,
            )


            def sigmoid(z):
                return 1 / (1 + np.exp(-z))


            df = pd.read_csv(
                config.CAUSAL_DB_PATH,
                sep='\t',
                usecols=['chr', 'bp', 'rsid', 'ea_mode', 'nea_ref']
            ).drop_duplicates()
            df = df.loc[df['chr'] == int(wildcards.chrom)]
            df['variant'] = df.apply(
                lambda row: f"chr{row['chr']}_{row['bp']}_{row['ea_mode']}_{row['nea_ref']}",
                axis=1,
            )

            f = h5py.File(preds_file,'r')
            out_file = h5py.File(str(output),'w')

            for _, row in tqdm(df.iterrows(),total=len(df),miniters=1000):
                pos_enf = np.array(f['pos'])

                i = np.where(pos_enf == row['bp'])
                sad = f['SAD'][i]

                if len(sad) < 1:
                    vep = 0
                else:
                    vep = np.abs(sad[0]).max()
                    vep = (sigmoid(vep) - .5) * 2

                dset_name = f"{row['rsid']}-{row['variant']}"
                if dset_name not in out_file:
                    out_file.create_dataset(dset_name,data=np.concatenate([[vep], [vep]]))

            preds_file.unlink()


rule all_get_enformer_1000_genomes_scores_for_chromosome:
    input:
        [results_root + "model_predictions_enformer_1000_genomes/model_predictions_chr" + str(chrom) + ".hdf5"
         for chrom in range(1,23)]


rule prepare_phenotype:
    input:
        results_root + "popsims/{dataset}/score/{dataset}_popsimilarity.txt.gz"
    output:
        results_root + "phenotypes/{dataset}/{ukb_field_id}/preprocessed.tsv.gz",
        results_root + "phenotypes/{dataset}/{ukb_field_id}/results.tsv"
    log:
        stdout=results_root + "phenotypes/{dataset}/{ukb_field_id}/preprocessed.log",
        stderr=results_root + "phenotypes/{dataset}/{ukb_field_id}/preprocessed.stderr"
    resources:
        partition='cpu-batch',
        time='3-00:00:00',
        threads=2,
        mem='16gb'
    run:
        with redirect_output(log):
            pgs.prepare_phenotype(
                ukb_field_id=wildcards.ukb_field_id,
                phenotypes_path=config.PHENOTYPES_PATH,
                pop_similarities=input[0],
                output_file=output[0],
                results_file=output[1],
                is_classification=not (
                        str(wildcards.ukb_field_id).endswith('-0.0') or
                        str(wildcards.ukb_field_id).endswith('-2.0')
                ),
            )


rule create_pgs_ct_weights_secondary_signals:
    input:
        [results_root + "model_predictions/{model_id}/model_predictions_chr" + str(i) + ".hdf5" for i in range(1,23)]
    output:
        [
            results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/CT_{pval_type}_" + str(p_val) + ".txt.gz"
            for p_val in config.P_VAL_RANGES]
    log:
        stdout=results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/ct_{pval_type}.log",
        stderr=results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/ct_{pval_type}.stderr"
    resources:
        partition='cpu-batch',
        time='24:30:00',
        mem='32gb',
        threads=2
    run:
        with redirect_output(log):
            pgs.create_pgs_ct_weights(
                causal_db_path=config.CAUSAL_DB_PATH,
                df_variants_path=config.DF_VARIANTS_PATH,
                meta_id=wildcards.meta_id,
                model_id=wildcards.model_id,
                pval_type=wildcards.pval_type,
                min_variants_coverage=wildcards.min_variants_coverage,
                out_dir=Path(output[0]).parent,
                model_preds_files=input,
                trait=get_trait_for_meta_id(wildcards.meta_id),
                p_val_range=config.P_VAL_RANGES,
                plink_path=config.PLINK_PATH,
                ld_ref=config.LD_REF,
                use_secondary_signals=True,
                threads=resources['threads'],
            )

rule compute_scores_for_pgs_secondary_signals:
    input:
        results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/{pgs_id}.txt.gz"
    output:
        results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/{pgs_id}/{dataset}/score/aggregated_scores.txt.gz"
    log:
        stdout=results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/{pgs_id}/{dataset}/{dataset}_pgs.log",
        stderr=results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/{pgs_id}/{dataset}/{dataset}_pgs.stderr",
    resources:
        # partition='cpu-batch,gpu-batch',
        partition='cpu-batch',
        time='03:00:00',
        threads=16,
        mem='192gb'
    retries: 3
    run:
        with redirect_output(log):
            out_dir = Path(output[0]).parent.parent.parent
            work_dir = out_dir / 'work'
            if work_dir.exists():
                print(f'Trying to remove {work_dir}')
                shell(f"rm -rf {work_dir}")

            cmd = f"(nextflow run pgscatalog/pgsc_calc -profile conda,mamba --target_build GRCh37 "
            cmd += f"--scorefile {input[0]} "
            cmd += f"--input datasets/{wildcards.dataset}.csv "
            cmd += f"--outdir {out_dir} "
            cmd += f"--genotypes_cache {results_root}genotypes_cache/ "
            cmd += "-c nextflow.config "
            cmd += "--min_overlap 0.3 "
            cmd += f"--max_cpu-batchs {resources['threads']} "
            cmd += f"--max_memory {resources['mem'].replace('gb','.GB')} "
            cmd += f"--process.cpu-batchs {resources['threads']} "
            cmd += f"--process.memory {resources['mem'].replace('gb','.GB')} "
            cmd += "--process.time 72.h "
            cmd += f"-work-dir {work_dir} "
            cmd += f") &> {log['stdout']}"
            print(f'Running cmd:\n{cmd}')
            shell(cmd)
            shell(f"rm -rf {work_dir}")

rule evaluate_score_secondary_signals:
    input:
        pgs_scores=results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/{pgs_id}/{dataset}/score/aggregated_scores.txt.gz",
        trait_file=results_root + "phenotypes/{dataset}/{ukb_field_id}/preprocessed.tsv.gz",
    output:
        results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/{pgs_id}/{dataset}/{ukb_field_id}/results.tsv",
    log:
        stdout=results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/{pgs_id}/{dataset}/{ukb_field_id}/results.log",
        stderr=results_root + "pgs_secondary/{meta_id}/{model_id}/{min_variants_coverage}/{pgs_id}/{dataset}/{ukb_field_id}/results.stderr",
    resources:
        partition='cpu-batch',
        # partition='cpu-batch,gpu-batch',
        time='01:00:00',
        threads=4,
        mem='32gb'
    run:
        with redirect_output(log):
            results = pgs.evaluate_score(
                scores_path=input['pgs_scores'],
                phenotypes_path=input['trait_file'],
                baseline_results_path=str(input['trait_file']).replace('preprocessed.tsv.gz','results.tsv'),
                # TODO a proper solution, e.g., with regex
                is_classification=not (
                        str(wildcards.ukb_field_id).endswith('-0.0') or
                        str(wildcards.ukb_field_id).endswith('-2.0')
                ),
            )
            results['model_id'] = wildcards.model_id
            results['pgs_id'] = wildcards.pgs_id
            results['dataset'] = wildcards.dataset
            results['ukb_field_id'] = wildcards.ukb_field_id
            results['min_variants_coverage'] = wildcards.min_variants_coverage
            results.to_csv(
                output[0],
                index=False,
                sep='\t',
            )

rule all_evaluate_scores_secondary_signals:
    input:
        dl_results=expand(
            expand(
                results_root + "pgs_secondary/{{meta_id}}/{model_id}/{min_variants_coverage}/{pgs_id}/{dataset}/{{ukb_field_id}}/results.tsv",
                min_variants_coverage=config.MIN_VARIANTS_COVERAGES,
                model_id=list(DL_MODEL_IDS_SECONDARY) + [
                    'enformer_normalized',
                    'cadd',
                ] + [
                    f'{model_name}_top{topk}_cws'
                    for model_name in ['enformer_normalized', 'basenji2_normalized', 'cadd'] for topk in [5, 10]
                ],
                pgs_id=DL_PGS_IDS,
                dataset=config.DATASETS,
            ),
            zip,
            meta_id=config.df_meta['meta_id'].values,
            ukb_field_id=config.df_meta['FieldID'].values,
        ),
        baseline_results=expand(
            expand(
                results_root + "pgs_secondary/{{meta_id}}/{model_id}/{min_variants_coverage}/{pgs_id}/{dataset}/{{ukb_field_id}}/results.tsv",
                min_variants_coverage=config.MIN_VARIANTS_COVERAGES,
                model_id=['all', 'cadd', 'sei'] + list(config.FINEMAPPING_TOOLS) + [
                    f'{model_name}_top{topk}_cws'
                    for model_name in ['cadd', 'sei', 'enformer_normalized', 'all'] for topk in [5, 10]
                ],
                pgs_id=BASELINE_PGS_IDS,
                dataset=config.DATASETS,
            ),
            zip,
            meta_id=config.df_meta['meta_id'].values,
            ukb_field_id=config.df_meta['FieldID'].values,
        )


rule run_permutation_test_secondary_signals:
    input:
        pgs_scores_1=results_root + "pgs_secondary/{meta_id}/{model_id_1}/{min_variants_coverage}/{pgs_id_1}/{dataset}/score/aggregated_scores.txt.gz",
        pgs_scores_2=results_root + "pgs_secondary/{meta_id}/{model_id_2}/{min_variants_coverage}/{pgs_id_2}/{dataset}/score/aggregated_scores.txt.gz",
        trait_file=results_root + "phenotypes/{dataset}/{ukb_field_id}/preprocessed.tsv.gz",
    output:
        results_root + "permutation_tests_secondary/{model_id_1}/{pgs_id_1}/{model_id_2}/{pgs_id_2}/{meta_id}/{min_variants_coverage}/{dataset}/{ukb_field_id}/results.tsv",
    log:
        stdout=results_root + "permutation_tests_secondary/{model_id_1}/{pgs_id_1}/{model_id_2}/{pgs_id_2}/{meta_id}/{min_variants_coverage}/{dataset}/{ukb_field_id}/results.log",
        stderr=results_root + "permutation_tests_secondary/{model_id_1}/{pgs_id_1}/{model_id_2}/{pgs_id_2}/{meta_id}/{min_variants_coverage}/{dataset}/{ukb_field_id}/results.stderr",
    resources:
        partition='cpu-batch',
        time='3-00:00:00',
        threads=4,
        mem='32gb'
    run:
        with redirect_output(log):
            results = pgs.evaluate_scores_with_permutation_test(
                scores_path_1=input['pgs_scores_1'],
                scores_path_2=input['pgs_scores_2'],
                phenotypes_path=input['trait_file'],
                baseline_results_path=str(input['trait_file']).replace('preprocessed.tsv.gz','results.tsv'),
                is_classification=not (
                        str(wildcards.ukb_field_id).endswith('-0.0') or
                        str(wildcards.ukb_field_id).endswith('-2.0')
                ),
                n_perms=config.N_PERMS_TESTS,
            )

            results['model_id_1'] = wildcards.model_id_1
            results['model_id_2'] = wildcards.model_id_2
            results['pgs_id_1'] = wildcards.pgs_id_1
            results['pgs_id_2'] = wildcards.pgs_id_2
            results['dataset'] = wildcards.dataset
            results['ukb_field_id'] = wildcards.ukb_field_id
            results['min_variants_coverage'] = wildcards.min_variants_coverage
            results.to_csv(
                output[0],
                index=False,
                sep='\t',
            )


rule all_run_permutation_tests_secondary_signals:
    input:
        expand(
            expand(
                expand(
                    results_root + "permutation_tests_secondary/{model_id_1}/{pgs_id_1}/{{model_id_2}}/{{pgs_id_2}}/{{{{meta_id}}}}/{min_variants_coverage}/{dataset}/{{{{ukb_field_id}}}}/results.tsv",
                    min_variants_coverage=config.MIN_VARIANTS_COVERAGES,
                    model_id_1=[
                        # your MIFM model ID goes here
                    ],
                    pgs_id_1=['CT_alt_1'],
                    dataset=config.DATASETS,
                ),
                zip,

                model_id_2=[
                    'all',
                    'all_top5_cws',
                    'all_top10_cws',

                    'basenji2',
                    'basenji2_top5_cws',
                    'basenji2_top10_cws',

                   'basenji2_normalized',
                   'basenji2_normalized_top5_cws',
                   'basenji2_normalized_top10_cws',

                    'cadd',
                    'cadd_top5_cws',
                    'cadd_top10_cws',

                    'sei',
                    'sei_top5_cws',
                    'sei_top10_cws',

                    'enformer_normalized',
                    'enformer_normalized_top5_cws',
                    'enformer_normalized_top10_cws',
                ] + list(config.FINEMAPPING_TOOLS),
                pgs_id_2=[
                             'CT_p_1', 'CT_p_1', 'CT_p_1', # all
                             'CT_alt_1','CT_alt_1','CT_alt_1', # basenji2
                             'CT_alt_1','CT_alt_1','CT_alt_1', # basenji2_normalized
                             'CT_alt_1','CT_alt_1','CT_alt_1', # enformer
                             'CT_p_1', 'CT_p_1', 'CT_p_1',  # cadd
                             'CT_p_1', 'CT_p_1', 'CT_p_1',  # sei

                         ] +
                         ['CT_p_1'] * len(config.FINEMAPPING_TOOLS),
            ),
            zip,
            meta_id=config.df_meta['meta_id'].values,
            ukb_field_id=config.df_meta['FieldID'].values,
        )


rule compute_attributions:
    output:
        results_root + "motifs/attributions/{model_id}/shap.npz",
        results_root + "motifs/attributions/{model_id}/ohe.npz"
    log:
        stdout=results_root + "motifs/attributions/{model_id}/shap.stdout",
        stderr=results_root + "motifs/attributions/{model_id}/shap.stderr"
    resources:
        partition='gpu-batchpro',
        time='3-00:00:00',
        gpus=1,
        mem='512gb',
        threads=16
    run:
        with redirect_output(log):
            motifs.compute_attributions(
                model_checkpoint=config.model_paths[wildcards.model_id],
                variants_path=config.CAUSAL_DB_PATH,
                neg_variants_path=config.NEG_DB_PATH,
                output_dir=results_root + '/motifs/attributions/',
                n_variants=30_000_000,
                n_neg_variants=100_000,
            )

rule compute_tf_modisco:
    input:
        shap=results_root + "motifs/attributions/{model_id}/shap.npz",
        ohe=results_root + "motifs/attributions/{model_id}/ohe.npz"
    output:
        results_root + "motifs/modisco/{model_id}/{num_seqlets}/modisco_results.h5"
    log:
        stdout=results_root + "motifs/modisco/{model_id}/{num_seqlets}/modisco_results.stdout",
        stderr=results_root + "motifs/modisco/{model_id}/{num_seqlets}/modisco_results.stderr"
    resources:
        partition='cpu-batch',
        time='3-00:00:00',
        mem='512gb',
        threads=8
    run:
        with redirect_output(log):
            out_dir = Path(output[0]).parent.parent.parent

            cmd = f"(modisco motifs "
            cmd += f"-s {input['ohe']} "
            cmd += f"-a {input['shap']} "
            cmd += f"-o {output[0]} "
            cmd += f"-n {wildcards.num_seqlets} "
            cmd += f") &> {log['stdout']}"
            print(f'Running cmd:\n{cmd}')
            shell(cmd)

rule generate_tf_modisco_report:
    input:
        results_root + "motifs/modisco/{model_id}/modisco_results.h5"
    output:
        results_root + "motifs/modisco/{model_id}/report/motifs.html"
    log:
        stdout=results_root + "motifs/modisco/{model_id}/report/report.stdout",
        stderr=results_root + "motifs/modisco/{model_id}/report/report.stderr"
    resources:
        partition='cpu-batch',
        time='3-00:00:00',
        mem='64gb',
        threads=2
    run:
        with redirect_output(log):
            out_dir = Path(output[0]).parent

            cmd = f"(modisco report "
            cmd += f"-i {input[0]} "
            cmd += f"-o {out_dir}/ "
            cmd += f"-s {out_dir}/ "
            if config.MODISCO_MOTIF_DB is not None:
                cmd += f"-m {config.MODISCO_MOTIF_DB} "
            cmd += f") &> {log['stdout']}"
            print(f'Running cmd:\n{cmd}')
            shell(cmd)

rule evaluate_enhancer_atlas:
    input:
        results_root + "model_predictions/{model_id}/predictions.ok"
    output:
        results_root + "enhancer_atlas/{model_id}/plots/plots.ok"
    log:
        stdout=results_root + "enhancer_atlas/{model_id}/plots/plots.log",
        stderr=results_root + "enhancer_atlas/{model_id}/plots/plots.stderr"
    resources:
        partition='cpu-batch',
        time='3-00:00:00',
        mem='128gb',
        threads=2
    run:
        with redirect_output(log):
            enhancer_atlas.main(
                variants_path=config.CAUSAL_DB_PATH,
                model_predictions_dir=Path(input[0]).parent,
                out_dir=Path(output[0]).parent,
            )
            open(str(output),'a').close()

rule all_evaluate_enhancer_atlas:
    input:
        expand(results_root + "enhancer_atlas/{model_id}/plots/plots.ok",model_id=DL_MODEL_IDS),


rule preprocess_genostan_analysis_data:
    output:
        results_root + "credible_set_genostan_binarized_model_scores_a7a0q7ar.pkl",
    log:
        stdout=results_root + "credible_set_genostan_binarized_model_scores_a7a0q7ar.stdout",
        stderr=results_root + "credible_set_genostan_binarized_model_scores_a7a0q7ar.stderr",
    resources:
        partition='cpu-batch',
        time='3-00:00:00',
        mem='512gb',
        threads=8
    run:
        with redirect_output(log):
            df = pd.read_pickle('results/credible_set_genostan_model_scores_a7a0q7ar.pkl')

            cell_lines = [c for c in df.columns if 'state' in c]

            state_names = sorted([
                'Elon.7',
                'ElonW.13',
                'ElonW.15',
                'Enh.6',
                'EnhW.8',
                'EnhWF.9',
                "Gen5'.16",
                'Low.12',
                'Low.17',
                'Low.18',
                'Low.20',
                'Low.3',
                'Low.4',
                'Low.5',
                'Prom.1',
                'Prom.19',
                'Repr.14',
                'Repr.2',
                'ReprEnh.11',
                'ReprW.10'
            ])

            state_subsets = {
                'Enh': ['Enh.6', 'EnhW.8', 'EnhWF.9', ],
                'Prom': ['Prom.1', 'Prom.19', ],
                'Repr': ['Repr.14', 'Repr.2', 'ReprW.10', ],
                'Elon': ['Elon.7', 'ElonW.13', 'ElonW.15', "Gen5'.16", ],
                'ReprEnh': ['ReprEnh.11', ],
                'Low': [
                    'Low.12',
                    'Low.17',
                    'Low.18',
                    'Low.20',
                    'Low.3',
                    'Low.4',
                    'Low.5',
                ],
            }

            all_state_names = state_names + list(state_subsets.keys())
            for c in tqdm(cell_lines):
                df[c] = df[c].apply(lambda s: '' if pd.isna(s) else state_names[int(s) - 1])


            def binarize_states(row):
                labels = {}
                for s in state_names:
                    s_count = 0
                    labels[s] = False
                    for c in cell_lines:
                        s_count += int(row[c] == s)
                        if s_count == 5:
                            labels[s] = True
                            break

                for state_subset_name, state_subset in state_subsets.items():
                    s_count = 0
                    labels[state_subset_name] = False
                    for c in cell_lines:
                        s_count += int(any(row[c] == s for s in state_subset))
                        if s_count == 5:
                            labels[state_subset_name] = True
                            break

                return pd.Series(labels)


            tqdm.pandas()
            df[all_state_names] = df.progress_apply(binarize_states,axis=1)
            df.to_pickle('results/credible_set_genostan_binarized_model_scores_a7a0q7ar.pkl')


rule preprocess_credible_sets_data:
    output:
        results_root + "credible_set_min_5_variants_a7a0q7ar_results.tsv",
    log:
        stdout=results_root + "credible_set_min_5_variants_a7a0q7ar_results.stdout",
        stderr=results_root + "credible_set_min_5_variants_a7a0q7ar_results.stderr",
    resources:
        partition='cpu-batch',
        time='3-00:00:00',
        mem='512gb',
        threads=8
    run:
        with redirect_output(log):
            from scipy import stats

            MIN_CRED_SET_SIZE = 5

            ea_col_name = 'ea_mode'
            nea_col_name = 'nea_ref'
            FINEMAPPING_TOOLS = (
                'abf',
                'susie',
                'finemap',
                'paintor',
                'caviarbf',
                'polyfun_finemap',
                'polyfun_susie',
            )
            df = pd.read_csv(
                'data/CAUSALdb/credible_set.txt',
                sep='\t',
                usecols=list(FINEMAPPING_TOOLS) + ['meta_id', 'block_id', 'rsid', 'chr', 'bp', ea_col_name,
                                                   nea_col_name],
            )
            df = df.drop_duplicates(['rsid', 'meta_id', 'block_id'])


            def modified_z_score(x):
                x = (x - np.median(x)) / stats.median_abs_deviation(x)
                return x[-1] > 3.5


            def IQR_method(x):
                Q1 = np.quantile(x,.25)
                Q3 = np.quantile(x,.75)
                IQR = Q3 - Q1

                return x[-1] > Q3 + 1.5 * IQR


            def get_model_score(row):
                preds_name = f"{row['rsid']}-chr{row['chr']}_{row['bp']}_{row[ea_col_name]}_{row[nea_col_name]}"
                h5_file = model_preds_files[int(row['chr'])]
                preds = h5_file[preds_name][0]
                score = np.abs(preds)

                return score


            model_preds_files = {
                int(Path(model_preds_file).name.split('chr')[1].replace('.hdf5','')): h5py.File(
                    model_preds_file,
                    'r',
                    #libver='latest',
                )
                for model_preds_file in Path('results/model_predictions/a7a0q7ar').glob('*.hdf5')
            }

            rows = []

            for _, group in tqdm(df.groupby(['meta_id', 'block_id']),total=len(df.groupby(['meta_id', 'block_id']))):
                if len(group) > MIN_CRED_SET_SIZE:
                    group['model_score'] = group.apply(get_model_score,axis=1)
                    for tool in list(FINEMAPPING_TOOLS) + ['model_score']:
                        scores = np.sort(group[tool])

                        rows.append({
                            'modified_z_score': modified_z_score(scores),
                            'IQR_method': IQR_method(scores),
                            'kurtosis': stats.kurtosis(scores),
                            'tool': tool,
                        })

            res = pd.DataFrame(rows)
            res.to_csv(
                str(output[0]),
                sep='\t',
                index=False,
            )

rule add_silencerDB_data:
    output:
        results_root + "credible_set_genostan_binarized_model_scores_silencerDB_a7a0q7ar.pkl",
    log:
        stdout=results_root + "credible_set_genostan_binarized_model_scores_silencerDB_a7a0q7ar.stdout",
        stderr=results_root + "credible_set_genostan_binarized_model_scores_silencerDB_a7a0q7ar.stderr",
    resources:
        partition='cpu-batch',
        time='3-00:00:00',
        mem='512gb',
        threads=8
    run:
        with redirect_output(log):
            df = pd.read_pickle('results/credible_set_genostan_binarized_model_scores_a7a0q7ar.pkl')

            df_silencerDB = pd.read_csv(
                'data/silencerDB/Homo_sapiens.bed',
                sep='\t',
                header=None,
                names=['chr', 'start', 'end', '1', '2', '3', '4', '5', '6', '7'],
            )
            df_silencerDB['chr'] = df_silencerDB['chr'].str.replace('chr','')
            df_silencerDB = df_silencerDB.loc[~df_silencerDB['chr'].isin(('X', 'Y', 'M'))]
            df_silencerDB['chr'] = df_silencerDB['chr'].astype(int)

            df['silencerDB'] = False

            #for _, row in tqdm(df_silencerDB.iterrows(), total=len(df_silencerDB), miniters=1000):
            for _, row in df_silencerDB.iterrows():
                df.loc[
                    (df['chr'] == row['chr'])
                    & (df['bp'].between(row['start'],row['end'])),
                    'silencerDB'
                ] = True

            df.to_pickle(str(output[0]))
