import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tangermeme.deep_lift_shap import deep_lift_shap

from src import dataloading, utils


def compute_attributions(
        model_checkpoint,
        variants_path,
        neg_variants_path,
        output_dir,
        batch_size=64,
        n_variants=1_000_000,
        n_neg_variants=100_000,
):
    model_checkpoint = Path(model_checkpoint)
    model = utils.load_model(model_checkpoint)

    df = pd.read_csv(
        variants_path,
        sep='\t',
    ).drop_duplicates('rsid')
    df = df.sample(n=min(n_variants, len(df)))

    df_neg = pd.read_csv(
        neg_variants_path,
        sep='\t',
    )
    df_neg = df_neg.sample(n=min(n_neg_variants, len(df_neg)))
    df_neg['ea_mode'] = df_neg['ea']
    df_neg['nea_ref'] = df_neg['nea']

    df = pd.concat([df, df_neg])

    datamodule = dataloading.VariantBlockDataModule(
        batch_size=batch_size,
        seq_len=model.hparams.seq_len,
        n_sampled_negative=0,
        encode_alt=False,
        predict_all=True,
        variants_df=df,
    )

    X_attrs = []
    Xs = []
    variant_names = []
    dloader = datamodule.predict_dataloader()
    for batch in tqdm(dloader, total=len(dloader)):
        x, _, _, block_infos = batch

        x = torch.stack([X for X in x if X.sum() == X.shape[-1]])

        X_attr = deep_lift_shap(
            model,
            x,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=0,
            warning_threshold=.9,
        )
        X_attrs.append(X_attr)
        Xs.append(x)
        for block_info in block_infos:
            for v in block_info['variant'].values:
                variant_names.append(v)

    X_attrs = torch.cat(X_attrs).detach().cpu().numpy()
    Xs = torch.cat(Xs).detach().cpu().numpy()
    variant_names = np.array(variant_names)

    output_dir = Path(output_dir) / model_checkpoint.parent.parent.name
    output_dir.mkdir(exist_ok=True, parents=True)
    np.savez(output_dir / 'shap.npz', X_attrs)
    np.savez(output_dir / 'ohe.npz', Xs)
    np.savez(output_dir / 'variant_names.npz', variant_names)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_checkpoint', type=str, required=True)
    p.add_argument('--variants_path', type=str, default='data/CAUSALdb/credible_set_10k.txt')
    p.add_argument('--neg_variants_path', type=str, default='data/CAUSALdb/neg_set_10k.txt')
    p.add_argument('--output_dir', type=str, default='results/motifs/')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--n_variants', type=int, default=1000 * 1000)
    p.add_argument('--n_neg_variants', type=int, default=100_000)
    args = p.parse_args()

    compute_attributions(**vars(args))
