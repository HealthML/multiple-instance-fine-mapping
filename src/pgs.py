import gzip
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression, LogisticRegression
import sklearn.metrics as sklearn_metrics
from tqdm import tqdm

FINEMAPPING_TOOLS = [
    'abf',
    'susie',
    'finemap',
    'paintor',
    'caviarbf',
    'polyfun_finemap',
    'polyfun_susie',
]




def prepend_lines_to_file(filepath, lines):
    lines = [l + '\n' for l in lines]
    with open(filepath, 'r') as file:
        existing_lines = file.readlines()

    # Combine the new lines with the existing lines
    all_lines = lines + existing_lines

    # Write the combined lines back to the file
    with open(filepath, 'w') as file:
        file.writelines(all_lines)


def gzip_file(filepath):
    with open(filepath, 'rb') as f_in:
        with gzip.open(str(filepath) + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def clump_sumstats(
        plink_path,
        ld_ref,
        sumstats,
        clumpfile,
        threads,
        pval_type,
):
    if len(ld_ref) == 1:
        # single ld ref file for all chromosomes
        subprocess.run(' '.join([
            plink_path,
            f'--bfile {ld_ref}',
            f'--clump-p1 1',
            f'--clump-p2 1',
            f'--clump-r2 0.1',
            f'--clump-kb 250',
            f'--clump {sumstats}',
            f'--clump-snp-field SNP_ALT_ID',
            f'--clump-field p',
            f'--threads {threads}',
            f'--out {clumpfile}',
        ]), shell=True)
        return clumpfile.with_suffix('.clumped')
    else:
        chromosomes = pd.read_csv(sumstats, sep='\t', usecols=['chr'])['chr'].unique()
        clumps = []
        for chrom in chromosomes:
            clumpfile = clumpfile.parent / f'chr{chrom}'
            subprocess.run(' '.join([
                plink_path,
                f'--bfile {ld_ref[str(chrom)]}',
                f'--clump-p1 1',
                f'--clump-p2 1',
                f'--clump-r2 0.1',
                f'--clump-kb 250',
                f'--clump {sumstats}',
                f'--clump-snp-field SNP_ALT_ID',
                f'--clump-field p',
                f'--threads {threads}',
                f'--out {clumpfile}',
            ]), shell=True)
            try:
                clumps.append(
                    pd.read_csv(
                        clumpfile.with_suffix('.clumped'),
                        delim_whitespace=True,
                    ))
            except FileNotFoundError:
                print(f'No clump results for chrom {chrom}')
        out_file = clumpfile.parent / f'{pval_type}.clumped'
        pd.concat(clumps).drop_duplicates().to_csv(out_file, index=False, sep=' ')

        return out_file


def create_pgs_ct_weights(
        causal_db_path,
        df_variants_path,
        meta_id,
        model_id,
        out_dir,
        model_preds_files,
        trait,
        p_val_range,
        plink_path,
        ld_ref,
        threads,
        pval_type,
        use_secondary_signals=False,
        ea_col_name='ea_mode',
        nea_col_name='nea_ref',
        min_variants_coverage=0,
):
    df_variants = pd.read_csv(
        df_variants_path,
        sep='\t',
    )
    df = pd.read_csv(
        causal_db_path,
        sep='\t',
        usecols=[
                    ea_col_name,
                    nea_col_name,
                    'ea',
                    'nea',
                    'rsid',
                    'chr',
                    'bp',
                    'p',
                    'beta',
                    'meta_id',
                    'block_id',
                    'primary',
                ] + list(FINEMAPPING_TOOLS)
    )
    df = df.loc[df['meta_id'] == meta_id]

    if len(df) < 1:
        raise Exception(f'Empty dataframe for meta_id {meta_id}')

    out_dir.mkdir(parents=True, exist_ok=True)

    if len(ld_ref) > 1:
        # for the UKB imputed data
        if 'ID' in df_variants:
            df['ID_REF'] = df.apply(
                lambda row: f"{row['chr']}:{row['bp']}{row['nea']}_{row['ea']}",
                axis=1,
            )
            df['ID_ALT'] = df.apply(
                lambda row: f"{row['chr']}:{row['bp']}{row['ea']}_{row['nea']}",
                axis=1,
            )
            # reference matches UKB
            df_ref = df.loc[df['ID_REF'].isin(df_variants['ID'])]
            df_ref['SNP_ALT_ID'] = df_ref['ID_REF']
            # reference flipped with UKB
            df_alt = df.loc[df['ID_ALT'].isin(df_variants['ID'])]
            df_alt['SNP_ALT_ID'] = df_alt['ID_ALT']
            df_full = df
            df = pd.concat([df_ref, df_alt]).drop_duplicates()

            blocks = []
            for block_id, block in df.groupby('block_id'):
                print(len(block) / len(df_full.loc[df_full['block_id'] == block_id]))
                if len(block) / len(df_full.loc[df_full['block_id'] == block_id]) >= float(min_variants_coverage):
                    blocks.append(block)
            df = pd.concat(blocks)
        else:
            df['SNP_ALT_ID'] = df.apply(
                lambda row: f"{row['chr']}:{row['bp']}{row['ea']}_{row['nea']}",
                axis=1,
            )
            df = df.loc[df['SNP_ALT_ID'].isin(df_variants['SNP_ALT_ID'])]
            df = df.loc[df['rsid'].isin(df_variants['rsid'])]
    else:
        df['SNP_ALT_ID'] = df['rsid']

    if not model_id.startswith('all'):
        if model_id in FINEMAPPING_TOOLS:
            df['p'] = 1 - df[model_id]
        elif model_id.split('_top')[0] in FINEMAPPING_TOOLS:
            df['p'] = 1 - df[model_id.split('_top')[0]]
        else:
            model_preds_files = {
                int(Path(model_preds_file).name.split('chr')[1].replace('.hdf5', '')):
                    h5py.File(model_preds_file, 'r') for model_preds_file in model_preds_files
            }

            def _get_model_p_val(row):
                preds_name = f"{row['rsid']}-chr{row['chr']}_{row['bp']}_{row[ea_col_name]}_{row[nea_col_name]}"
                h5_file = model_preds_files[int(row['chr'])]

                if any (m in model_id for m in ('cadd', 'sei')):
                    try:
                        if preds_name not in h5_file:
                            preds_name_short = f"chr{row['chr']}_{row['bp']}_{row[ea_col_name]}_{row[nea_col_name]}"
                            preds_name_short_alt = f"chr{row['chr']}_{row['bp']}_{row[nea_col_name]}_{row[ea_col_name]}"
                            preds_name = [k for k in list(h5_file.keys()) if (preds_name_short in k or preds_name_short_alt in k)][0]
                        preds = h5_file[preds_name][0] if 'alt' in pval_type else h5_file[preds_name][1]
                    except (KeyError, IndexError):
                        print(f'{preds_name} not found')
                        preds = 0

                elif model_id == 'enformer':
                    try:
                        preds = h5_file[preds_name][0] if 'alt' in pval_type else h5_file[preds_name][1]
                    except KeyError:
                        preds = 0
                else:
                    preds = h5_file[preds_name][0] if 'alt' in pval_type else h5_file[preds_name][1]

                pval = np.abs(preds)

                # plink seems to ignore variants with p-val equal to 1
                max_pval = .999999999
                pval = min(max_pval - pval, max_pval)

                if '_x_pval' in pval_type:
                    pval = pval * row['p']

                return pval

            df['p_gwas'] = df['p']
            df['p'] = df.apply(
                _get_model_p_val,
                axis=1,
            )
            if '_or_pval' in pval_type:
                # min probability of the DL model
                p_threshold = float(pval_type.split('_')[-1])
                groups = []
                for _, group in df.groupby('block_id'):
                    if group['p'].min() > p_threshold:
                        group['p'] = group['p_gwas']
                    groups.append(group)
                df = pd.concat(groups)

            if '_top_' in pval_type:
                # filter to top k variants of a finemapping tool
                top_k = int(pval_type.split('_')[-2])
                tool_name = pval_type.split('_')[-1]
                groups = []
                for _, group in df.groupby(['block_id', 'primary'] if use_secondary_signals else 'block_id'):
                    group = group.sort_values(tool_name, ascending=False).iloc[:top_k]
                    groups.append(group)
                df = pd.concat(groups)

    sumstats = out_dir / f'sumstats_{pval_type}.txt'
    df[['SNP_ALT_ID', 'p', 'rsid', 'chr', 'bp', 'ea', 'nea', 'beta']].to_csv(
        sumstats,
        sep='\t',
        index=False,
    )

    for p_val_threshold in p_val_range:
        df_clump_p = df.loc[df['p'] <= p_val_threshold]

        # handle the top-k finemapping variants separately:
        if '_top' in model_id and '_cws' in model_id:
            # common weighting scheme
            topk = int(model_id.split('_top')[1].split('_cws')[0])
            dfs = []
            for _, df_group in df_clump_p.groupby(['block_id', 'primary'] if use_secondary_signals else 'block_id'):
                df_block_ = df_group.sort_values('p').iloc[:topk].copy()
                # compute adjusted betas
                df_block_['beta'] = df_block_['beta'] / len(df_block_)
                dfs.append(df_block_)
            df_clump_p = pd.concat(dfs)
        elif '_top' in model_id:
            topk = int(model_id.split('_top')[1])
            dfs = []
            for _, df_group in df_clump_p.groupby(['block_id', 'primary'] if use_secondary_signals else 'block_id'):
                df_block_ = df_group.sort_values('p').iloc[:topk].copy()
                # compute adjusted betas
                df_block_ = get_adjusted_betas(df_block_, 'data/LD_panels/1000_genomes/ldblk_1kg_eur/')
                df_block_['beta'] = df_block_['beta_adjusted']
                dfs.append(df_block_)
            df_clump_p = pd.concat(dfs)
        else:
            df_clump_p = df_clump_p.loc[
                df_clump_p.groupby(
                    ['block_id', 'primary'] if use_secondary_signals else 'block_id'
                )['p'].idxmin()
            ]

        df_clump_p = df_clump_p.rename(columns={
            'chr': 'chr_name',
            'bp': 'chr_position',
            'ea': 'effect_allele',
            'nea': 'other_allele',
            'beta': 'effect_weight',
        })[['chr_name', 'chr_position', 'effect_allele', 'other_allele', 'effect_weight']]
        pgs_path = out_dir / f'CT_{pval_type}_{p_val_threshold}.txt'
        df_clump_p.to_csv(
            pgs_path,
            sep='\t',
            index=False,
        )
        prepend_lines_to_file(
            pgs_path,
            [
                f'#pgs_name=CT_{pval_type}_{p_val_threshold}',
                f'#trait_reported={trait}',
                '#genome_build=GRCh37',
            ]
        )
        gzip_file(pgs_path)
        os.remove(pgs_path)


def create_pgs_ct_weights_from_clumps(
        clumps_path,
        out_filepath,
        pval_type,
        score_type,
        trait,
):
    df_clump = pd.read_csv(
        clumps_path,
        sep='\t',
    )
    df_clump = df_clump.loc[(df_clump['score_type'] == score_type) & (df_clump['pval_col'] == pval_type)]
    df_clump = df_clump.rename(columns={
        'CHR': 'chr_name',
        'BP': 'chr_position',
        'ea': 'effect_allele',
        'nea': 'other_allele',
        'beta': 'effect_weight',
    })[['chr_name', 'chr_position', 'effect_allele', 'other_allele', 'effect_weight']]
    df_clump.to_csv(
        out_filepath,
        sep='\t',
        index=False,
    )
    prepend_lines_to_file(
        out_filepath,
        [
            f'#pgs_name=CT_{pval_type}_{score_type}',
            f'#trait_reported={trait}',
            '#genome_build=GRCh37',
        ]
    )
    gzip_file(out_filepath)
    os.remove(out_filepath)


COVS = [
    '31-0.0',  # sex
    '21022-0.0',  # age
    '54-0.0',  # assessment center
    '22000-0.0',  # geno batch
]
GEN_PCS = [f'22009-0.{i}' for i in range(1, 11)]
N_FOLDS = 5


def prepare_phenotype(
        ukb_field_id,
        phenotypes_path,
        output_file,
        results_file,
        pop_similarities,
        is_classification,
):
    df = pd.read_csv(
        phenotypes_path,
        usecols=['eid', ukb_field_id] + COVS + GEN_PCS,
    )
    df = df.dropna()
    df['eid'] = df['eid'].astype(str)


    if pop_similarities is not None:
        df_pop = pd.read_csv(
            pop_similarities,
            sep='\t',
        )
        df_pop['IID'] = df_pop['IID'].astype(str)

        df = df.merge(df_pop, left_on='eid', right_on='IID', how='inner')
        df = df.rename(columns={'MostSimilarPop': 'pop'})
    else:
        df['pop'] = 'ALL'

    X = df[COVS + GEN_PCS].values
    y = df[ukb_field_id].values

    if not is_classification:
        # standardize the phenotype
        mu, std = y.mean(), y.std()
        y = (y - mu) / std
        # store them in the csv so we can reconstruct them
        df[['y_mu', 'y_std']] = mu, std

    lm = LinearRegression().fit(X=X, y=y)
    resid = y - lm.predict(X)
    df['y_resid'] = resid
    df['y'] = y

    results = []
    if str(ukb_field_id).endswith('-0.0'):
        seed = int(ukb_field_id.split('-')[0])
    else:
        seed = int.from_bytes(ukb_field_id[:4].encode('utf8'), 'little')
    rng = np.random.RandomState(seed=seed)

    for fold_idx in range(N_FOLDS):
        # create a random 2:1 split of train/test subjects
        while True:
            is_test = rng.binomial(1, .33, y.shape)
            df[f'is_test_fold_{fold_idx}'] = is_test
            if (
                    not is_classification or
                    # ensure there are always some cases present in each subsplit of data
                    df.groupby([f'is_test_fold_{fold_idx}', 'pop'])['y'].mean().min() > 0
            ):
                break

        df_train = df.loc[df[f'is_test_fold_{fold_idx}'] == 0]
        df_test = df.loc[df[f'is_test_fold_{fold_idx}'] == 1]
        results_fold = train_and_eval_lm(
            df_train=df_train,
            df_test=df_test,
            with_pgs=False,
            is_classification=is_classification,
        )
        results_fold['fold_idx'] = fold_idx
        results.append(results_fold)
    results_df = pd.concat(results).fillna(0)
    results_df = results_df.groupby(['pop', 'class_weight']).mean().reset_index().drop(labels='fold_idx', axis=1)

    results_df.to_csv(
        results_file,
        index=False,
        sep='\t',
    )
    df.to_csv(
        output_file,
        index=False,
        sep='\t',
    )


def cross_entropy_log_likelihood(
        prob_y,
        log_prob_y,
        y_true,
):
    # return -(y_true * log_prob_y).mean() - ((1 - y_true) * np.log(1 - np.clip(prob_y, a_min=0, a_max=.999999))).mean()
    return np.sum(-np.log(1 + prob_y)) + np.sum(y_true * log_prob_y)


def mcfadden_rsquare(
        prob_y,
        log_prob_y,
        prob_y_null,
        log_prob_y_null,
        y_true,
):
    likelihood = cross_entropy_log_likelihood(
        prob_y,
        log_prob_y,
        y_true
    )
    likelihood_null = cross_entropy_log_likelihood(
        prob_y_null,
        log_prob_y_null,
        y_true,
    )
    return 1.0 - likelihood / likelihood_null


def mcfadden_rsquare_adjusted(
        prob_y,
        log_prob_y,
        prob_y_null,
        log_prob_y_null,
        y_true,
        K,
):
    likelihood = cross_entropy_log_likelihood(
        prob_y,
        log_prob_y,
        y_true
    )
    likelihood_null = cross_entropy_log_likelihood(
        prob_y_null,
        log_prob_y_null,
        y_true,
    )
    return 1.0 - (likelihood - K) / likelihood_null


def auprc_score(y_true, y_score):
    p, r, _ = sklearn_metrics.precision_recall_curve(y_true, y_score)
    return sklearn_metrics.auc(r, p)


def eval_model(
        model,
        X,
        y,
        is_classification,
        model_null=None,
        y_hat=None,
        y_hat_null=None,
        only_ll=False,
):
    if is_classification:
        if y_hat is None:
            y_hat = model.predict_proba(X)[:, 1]
        log_prob_y = np.log(np.clip(y_hat, a_min=0.00000000001, a_max=1))
        ll = cross_entropy_log_likelihood(y_hat, log_prob_y, y)

        if only_ll and is_classification:
            return {'log_likelihood': ll}

        res = {
            'auroc_min_0_5': max(sklearn_metrics.roc_auc_score(y_true=y, y_score=y_hat), .5),
            'auroc': sklearn_metrics.roc_auc_score(y_true=y, y_score=y_hat),
            'auprc': auprc_score(y_true=y, y_score=y_hat),
            'accuracy': sklearn_metrics.accuracy_score(y_true=y, y_pred=y_hat > .5),
            'balanced_accuracy_min_0_5': max(sklearn_metrics.balanced_accuracy_score(y_true=y, y_pred=y_hat > .5), .5),
            'balanced_accuracy': sklearn_metrics.balanced_accuracy_score(y_true=y, y_pred=y_hat > .5),
            'f1': sklearn_metrics.f1_score(y_true=y, y_pred=y_hat > .5),
            'precision': sklearn_metrics.precision_score(y_true=y, y_pred=y_hat > .5),
            'recall': sklearn_metrics.recall_score(y_true=y, y_pred=y_hat > .5),
            'log_likelihood': ll,
            'y_hat': y_hat,
        }

        if model_null is not None or y_hat_null is not None:
            if y_hat_null is None:
                y_hat_null = model_null.predict_proba(X[:, :-1])[:, 1]
            log_prob_y_null = np.log(np.clip(y_hat_null, a_min=0.00000000001, a_max=1))
            ll_null = cross_entropy_log_likelihood(y_hat_null, log_prob_y_null, y)
            res['mcfadden_r2'] = 1 - ll / ll_null
            res['mcfadden_r2_min_0'] = max(1 - ll / ll_null, 0)
            res['y_hat_null'] = y_hat_null

        return res

    if y_hat is None:
        y_hat = model.predict(X)
    return {
        'r2': sklearn_metrics.r2_score(y_true=y, y_pred=y_hat),
        'r2_min_0': max(sklearn_metrics.r2_score(y_true=y, y_pred=y_hat), 0),
        'mse': sklearn_metrics.mean_squared_error(y_true=y, y_pred=y_hat),
        'mae': sklearn_metrics.mean_absolute_error(y_true=y, y_pred=y_hat),
        'y_hat': y_hat,
    }


def train_and_eval_lm(
        df_train,
        df_test,
        with_pgs,
        is_classification,
        return_predictions=False,
):
    cols = COVS + GEN_PCS
    if with_pgs:
        cols += ['SUM']

    results_rows = []
    for pop in pd.concat([df_train, df_test])['pop'].unique():
        X_train = df_train.loc[df_train['pop'] == pop][cols].values
        y_train = df_train.loc[df_train['pop'] == pop]['y'].values
        X_test = df_test.loc[df_test['pop'] == pop][cols].values
        y_test = df_test.loc[df_test['pop'] == pop]['y'].values

        # for class_weight in ('balanced', None):
        for class_weight in (None,):
            lm_null = None
            if not is_classification:
                if class_weight == 'balanced':
                    continue
                y = np.concatenate([y_train, y_test])
                mu_y, std_y = y.mean(), y.std()
                y_train = (y_train - mu_y) / std_y
                y_test = (y_test - mu_y) / std_y
                lm = LinearRegression().fit(X=X_train, y=y_train)
            else:
                C = 10
                penalty = 'l2'
                lm = LogisticRegression(
                    max_iter=1000,
                    C=C,
                    penalty=penalty,
                    class_weight=class_weight,
                ).fit(X=X_train, y=y_train)
                if with_pgs:
                    lm_null = LogisticRegression(
                        max_iter=1000,
                        C=C,
                        penalty=penalty,
                        class_weight=class_weight,
                    ).fit(X=X_train[:, :-1], y=y_train)

            results_pop = eval_model(
                model=lm,
                model_null=lm_null,
                X=X_test,
                y=y_test,
                is_classification=is_classification,
            )
            results_pop['pop'] = pop
            results_pop['class_weight'] = class_weight or 'uniform'

            if return_predictions:
                results_pop['y'] = y_test
            else:
                del results_pop['y_hat']
                if 'y_hat_null' in results_pop:
                    del results_pop['y_hat_null']

            if with_pgs:
                results_pop['beta_pgs'] = lm.coef_[0, -1] if is_classification else lm.coef_[-1]
            results_rows.append(results_pop)

    return pd.DataFrame(results_rows)


def evaluate_score(
        scores_path,
        phenotypes_path,
        baseline_results_path,
        is_classification,
        df_pgs=None,
        return_predictions=False,
        compute_baseline_diff=True,
):
    np.random.seed(1)

    if df_pgs is None:
        df_pgs = pd.read_csv(
            scores_path,
            sep='\t',
        )
        df_pgs['IID'] = df_pgs['IID'].astype(str)
    df_pheno = pd.read_csv(
        phenotypes_path,
        sep='\t',
    )
    df_pheno['eid'] = df_pheno['eid'].astype(str)
    df = df_pgs.merge(df_pheno, left_on='IID', right_on='eid', how='inner')

    results_baseline = pd.read_csv(
        baseline_results_path,
        sep='\t',
    )

    results = []
    for fold_idx in range(N_FOLDS):
        df_train = df.loc[df[f'is_test_fold_{fold_idx}'] == 0]
        df_test = df.loc[df[f'is_test_fold_{fold_idx}'] == 1]
        results_fold = train_and_eval_lm(
            df_train=df_train,
            df_test=df_test,
            with_pgs=True,
            is_classification=is_classification,
            return_predictions=True,
        )
        results_fold['fold_idx'] = fold_idx
        results.append(results_fold)
    results = pd.concat(results)
    predictions = results[['y_hat', 'y', 'pop', 'fold_idx'] + (['y_hat_null'] if is_classification else [])].copy()
    results = results.drop(columns=['y_hat', 'y_hat_null', 'y'], errors='ignore'
                           ).groupby(['pop', 'class_weight']).mean().reset_index().drop(labels='fold_idx', axis=1)
    results = results.merge(results_baseline, on=['pop', 'class_weight'], suffixes=['', '_baseline'])

    if compute_baseline_diff:
        for c in results.columns:
            if c == 'pop' or 'baseline' in c or c == 'class_weight' or 'beta_pgs' in c or 'mcfadden' in c:
                continue
            baseline_scores = results[f'{c}_baseline']
            results[f'{c}_diff'] = results[c] - baseline_scores

    if is_classification:
        res_unsupervised = unsupervised_classification_metrics(df)
        results = results.merge(res_unsupervised, on=['pop', 'class_weight'], how='left')

    if return_predictions:
        return results, predictions
    return results


def evaluate_scores_with_permutation_test(
        scores_path_1,
        scores_path_2,
        phenotypes_path,
        baseline_results_path,
        is_classification,
        n_perms,
        keep_eur=False,
):
    def warn(*args, **kwargs):
        pass

    import warnings
    warnings.warn = warn

    np.random.seed(1)

    scores_1 = pd.read_csv(scores_path_1, sep='\t')
    scores_1['IID'] = scores_1['IID'].astype(str)
    scores_2 = pd.read_csv(scores_path_2, sep='\t')
    scores_2['IID'] = scores_2['IID'].astype(str)

    if not keep_eur:
        df_pheno = pd.read_csv(phenotypes_path, sep='\t')
        iids_eur = df_pheno.loc[df_pheno['pop'] == 'EUR']['eid'].astype(str)
        scores_1 = scores_1.loc[~scores_1['IID'].isin(iids_eur)]
        scores_2 = scores_2.loc[~scores_2['IID'].isin(iids_eur)]
    scores_1, scores_2 = scores_1.sort_values('IID'), scores_2.sort_values('IID')

    assert scores_1['IID'].equals(scores_2['IID'])

    res_1, df_preds_1 = evaluate_score(
        scores_path=None,
        phenotypes_path=phenotypes_path,
        baseline_results_path=baseline_results_path,
        is_classification=is_classification,
        df_pgs=scores_1,
        return_predictions=True,
        compute_baseline_diff=False,
    )
    res_2, df_preds_2 = evaluate_score(
        scores_path=None,
        phenotypes_path=phenotypes_path,
        baseline_results_path=baseline_results_path,
        is_classification=is_classification,
        df_pgs=scores_2,
        return_predictions=True,
        compute_baseline_diff=False,
    )

    df_null = []
    for ((pop, fold_idx), df_group_1), (_, df_group_2) in tqdm(zip(
            df_preds_1.groupby(['pop', 'fold_idx']),
            df_preds_2.groupby(['pop', 'fold_idx']),
    ), total=df_preds_1['pop'].nunique() * df_preds_1['fold_idx'].nunique()):
        df_group_1, df_group_2 = df_group_1.iloc[0], df_group_2.iloc[0]
        assert df_group_1['pop'] == df_group_2['pop']
        assert df_group_1['fold_idx'] == df_group_2['fold_idx']

        y = df_group_1['y']
        y_hat_1 = df_group_1['y_hat']
        y_hat_2 = df_group_2['y_hat']

        y_hat_null = None
        if 'y_hat_null' in df_group_2:
            y_hat_null = df_group_2['y_hat_null']

        perms = np.random.randint(0, 2, size=(n_perms, len(y)))
        for perm_idx, perm in enumerate(perms):
            y_hat_perm = np.array([y_hat_1[i] if perm[i] == 0 else y_hat_2[i] for i in range(len(y))])

            res = eval_model(
                model=None,
                X=None,
                y=y,
                is_classification=is_classification,
                model_null=None,
                y_hat=y_hat_perm,
                y_hat_null=y_hat_null,
                # only_ll=True,
            )
            if 'y_hat' in res:
                del res['y_hat']
            if 'y_hat_null' in res:
                del res['y_hat_null']
            res['perm_idx'] = perm_idx
            res['pop'] = pop
            res['fold_idx'] = fold_idx
            res['class_weight'] = 'uniform'
            df_null.append(res)

    df_null = pd.DataFrame(df_null).groupby(['pop', 'perm_idx', 'class_weight']).mean().reset_index()

    df_null = df_null.loc[df_null['class_weight'] != 'balanced']
    res_1 = res_1.loc[res_1['class_weight'] != 'balanced']
    res_2 = res_2.loc[res_2['class_weight'] != 'balanced']

    metrics = ['log_likelihood', 'auroc', 'mcfadden_r2'] if is_classification else ['r2']
    rows = []
    for (pop, df_null_pop), (_, res_1_pop), (_, res_2_pop) in zip(
            df_null.groupby('pop'),
            res_1.groupby('pop'),
            res_2.groupby('pop'),
    ):
        for metric in metrics:
            diff_obs = (res_1_pop[metric] - res_2_pop[metric]).iloc[0]
            dists = pdist(np.expand_dims(df_null_pop[metric].values, 1), metric='minkowski', p=1)
            pval = (np.abs(diff_obs) <= dists).mean()
            # pval = ((np.abs(diff_obs) <= dists).sum() + 1) / (len(dists) + 1) # corrected, so that p-vals are never equal to 0
            rows.append({
                'pop': pop,
                'metric': metric,
                'pval': pval,
                'diff_obs': diff_obs,
            })

    return pd.DataFrame(rows).sort_values('metric')


def unsupervised_classification_metrics(df):
    results = []
    for pop, group in df.groupby('pop'):
        res = {
            'pop': pop,
            'corr': group[['SUM', 'y']].corr().values[0, 1],
            'class_weight': 'uniform',
        }

        base_freq = group['y'].mean()
        if base_freq == 0:
            base_freq = 1

        for quant in (.5, .6, .7, .8, .9, .95, .99):
            group_quant = group.loc[group['SUM'] >= group['SUM'].quantile(quant)]
            res[f'OR_{quant}'] = group_quant['y'].mean() / base_freq
        results.append(res)

    return pd.DataFrame(results)
