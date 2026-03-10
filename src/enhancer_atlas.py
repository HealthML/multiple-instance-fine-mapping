import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')


def get_stats_for_thresholds(df, col):
    n_blocks = df['block_id'].nunique()
    rows = []
    for thresh in np.linspace(0, .99, 100):
        df = df.loc[df[col] >= thresh]
        grouped = df.groupby('block_id')
        enh_peak_nunique = grouped['Enh_peak'].nunique()
        rsid_nunique = grouped['rsid'].nunique()
        row = {
            col: thresh,
            'enh_nunique_per_block': enh_peak_nunique.mean(),
            'enh_nunique_per_rsid_per_block': (enh_peak_nunique / rsid_nunique).mean(),
            'n_variants_per_block': rsid_nunique.sum() / n_blocks,
        }
        df_gr_i = grouped
        for i in range(1, 6):
            row[f'perc_blocks_with_{i}_variants'] = (rsid_nunique >= i).sum() / n_blocks

            df_gr_i = df_gr_i.filter(lambda gr: gr['rsid'].nunique() >= i).groupby('block_id')
            row[f'enh_nunique_per_rsid_with_{i}_variants'] = (
                    df_gr_i['Enh_peak'].nunique() / df_gr_i['rsid'].nunique()).mean()
            df_gr_eq_i = grouped.filter(lambda gr: gr['rsid'].nunique() == i).groupby('block_id')
            row[f'enh_nunique_with_{i}_variants'] = df_gr_eq_i['Enh_peak'].nunique().mean()

        rows.append(row)
    return pd.DataFrame(rows)


def main(
        variants_path,
        model_predictions_dir,
        out_dir,
        debug=False,
):
    def plt_show():
        if debug:
            plt.show()

    model_predictions_dir = Path(model_predictions_dir)
    assert model_predictions_dir.exists()
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    col_names = [
        'susie',
        'finemap',
        'paintor',
        'caviarbf',
        'polyfun_finemap',
        'polyfun_susie',
        'alt',
        'diff',
    ]

    df = pd.read_csv(
        variants_path,
        sep='\t',
    )
    rows = []
    for _, group in df.groupby('rsid'):
        group[col_names[:-2]] = group[col_names[:-2]].max()
        rows.append(group.iloc[0])
    df = pd.DataFrame(rows)

    def get_model_p_vals(row):
        path_preds = model_predictions_dir / f"{row['rsid']}-chr{row['chr']}_{row['bp']}_{row['ea_mode']}_{row['nea_ref']}.npz"

        preds = np.load(path_preds)
        preds_diff = preds['diff'][0]
        preds_alt = preds['alt'][0]

        return preds_alt, preds_diff

    df[['alt', 'diff']] = df.apply(
        get_model_p_vals,
        result_type="expand",
        axis=1,
    )

    rows = []
    for _, group in df.groupby('block_id'):
        rows.append({
            'n_variants': group['rsid'].nunique(),
            'n_enh': group['Enh_peak'].nunique(),
        })
    sns.lineplot(
        data=pd.DataFrame(rows),
        x='n_variants',
        y='n_enh',
        errorbar=None,
    )
    plt.savefig(out_dir / 'n_enh_x_wrt_variants.png')
    plt_show()
    plt.close()

    df_stats = get_stats_for_thresholds(df, col='alt')

    sns.lineplot(
        data=df_stats,
        x='alt',
        y='n_variants_per_block',
    )
    plt.ylim(0, None)
    plt.savefig(out_dir / 'n_variants_per_block.png')
    plt_show()
    plt.close()

    sns.lineplot(
        data=df_stats,
        x='alt',
        y='enh_nunique_per_block',
    )
    plt.ylim(0, None)
    plt.savefig(out_dir / 'enh_nunique_per_block.png')
    plt_show()
    plt.close()

    sns.lineplot(
        data=df_stats,
        x='alt',
        y='enh_nunique_per_rsid_per_block',
    )
    plt.savefig(out_dir / 'enh_nunique_per_rsid_per_block.png')
    plt_show()
    plt.close()

    sns.lineplot(
        data=pd.wide_to_long(
            df_stats,
            'perc_blocks_with_',
            i='alt',
            j='n_variants',
            suffix='\\d+\\_variants'
        ),
        x='alt',
        y=f'perc_blocks_with_',
        hue='n_variants',
    )
    plt.savefig(out_dir / 'perc_blocks_with_.png')
    plt_show()
    plt.close()

    plt.figure(figsize=(10, 5))
    g = sns.lineplot(
        data=pd.wide_to_long(
            df_stats,
            'enh_nunique_with_',
            i='alt',
            j='n_variants',
            suffix='\\d+\\_variants'
        ),
        x='alt',
        y=f'enh_nunique_with_',
        hue='n_variants',
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(out_dir / 'enh_nunique_with_.png')
    plt_show()
    plt.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variants_path', type=str, default='data/CAUSALdb/credible_set_100k.txt')
    p.add_argument('--model_predictions_dir', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    main(**vars(args))
