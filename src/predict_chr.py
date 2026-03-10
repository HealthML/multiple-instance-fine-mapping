import argparse
import gc
from pathlib import Path

import torch.nn.functional as F
import pandas as pd
import h5py
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from src import dataloading, utils

class PretrainedBasenji(pl.LightningModule):
    def __init__(self):
        super().__init__()

        from basenji2_pytorch import Basenji2, basenji2_params, basenji2_weights

        self.mus = None
        self.stds = None

        self.basenji2 = Basenji2(basenji2_params["model"])
        self.basenji2.load_state_dict(torch.load(basenji2_weights()))
        self.basenji2.eval()


    def predict_step(self, batch, batch_idx=None):
        x, _, _, block_infos = batch

        # preds = self.basenji2(x)
        preds = []
        batch_size = 128
        for i in range(0, x.shape[0], batch_size):
            preds.append(self.basenji2(x[i:i+batch_size]))
        preds = torch.cat(preds, dim=0)

        # normalize by precomputed moments
        if self.mus is not None:
            preds = (preds - self.mus) / self.stds

        preds_alt = preds[:preds.shape[0] // 2]
        preds_ref = preds[preds.shape[0] // 2:]

        res = (F.sigmoid((preds_alt - preds_ref).abs().max(2)[0].max(1, keepdims=True)[0]) - .5) * 2

        return (
            res,
            # return the same predictions for alt and diff
            res,
            pd.concat([b['variant'] for b in block_infos]).values,
            pd.concat([b['rsid'] for b in block_infos]).values,
        )

def compute_1000_genomes_moments(model, chrom):
    df_snps = pd.read_csv(
        'data/1000_genomes/snp141Common.txt.gz',
        sep='\t',
        header=None,
    )[[1, 2, 3, 4, 9, 20]].rename(columns={
        1: 'chr',
        2: 'bp',
        3: 'end',
        4: 'rsid',
        9: 'alleles',
        20: 'sources',
    })
    df_snps['chr'] = df_snps['chr'].str.replace('chr', '')
    df_snps = df_snps[
        (df_snps['sources'].str.contains('1000GENOMES'))
        & (df_snps['chr'].astype(str) == str(chrom))
        ]
    df_snps['nea'] = df_snps['alleles'].apply(lambda a: a.split('/')[0])
    df_snps['ea'] = df_snps['alleles'].apply(lambda a: a.split('/')[-1])
    df_snps = df_snps[df_snps['ea'].isin(list('ATCG'))]
    df_snps = df_snps[df_snps['nea'].isin(list('ATCG'))]

    df_snps = df_snps.sample(n=min(10_000, len(df_snps)))

    datamodule = dataloading.VariantBlockDataModule(
        batch_size=1,
        seq_len=16_513,
        n_sampled_negative=0,
        encode_alt=False,
        predict_all=True,
        predict_all_chrom=chrom,
        block_size_predict_all=128,
        variants_df=df_snps,
        use_genostan_labels=False,
        nea_col_name='nea',
        ea_col_name='ea',
    )

    #model = model.cuda()
    with torch.no_grad():
        # estimate means
        mus = np.zeros((5313,))
        for batch in tqdm(datamodule.predict_dataloader()):
            x = batch[0]#.cuda()
            res = model.basenji2(x)
            mus += (res.sum(dim=[0, 1]) / len(df_snps) / res.shape[1]).cpu().numpy()

        # estimate stds
        stds = np.zeros((5313,))
        for batch in tqdm(datamodule.predict_dataloader()):
            x = batch[0]#.cuda()
            res = model.basenji2(x)
            stds += (((res - mus) ** 2).sum(dim=[0, 1])  / len(df_snps) / res.shape[1]).cpu().numpy()
        stds = np.sqrt(stds)

    return mus, stds


def generate_predictions(
        model_checkpoint,
        variants_path,
        results_dir='results/model_predictions/',
        save_batch_size=8,
        batch_size=16,
        load_pretrained_enformer=False,
        load_pretrained_basenji=False,
        basenji_normalized=False,
        predict_all_chrom=None,
):
    if load_pretrained_basenji:
        model = PretrainedBasenji()

        if basenji_normalized:
            mus, stds = compute_1000_genomes_moments(model=model, chrom=predict_all_chrom)
            model.mus = mus
            model.stds = stds

        datamodule = dataloading.VariantBlockDataModule(
            batch_size=1,
            # seq_len=model.basenji2.seq_length,
            seq_len=16_513,
            n_sampled_negative=0,
            encode_alt=True,
            predict_all=True,
            predict_all_chrom=predict_all_chrom,
            variants_path=variants_path,
            use_genostan_labels=False,
        )
        results_dir = Path(results_dir) / ('basenji2_normalized' if basenji_normalized else 'basenji2')
        results_dir.mkdir(exist_ok=True, parents=True)
    else:
        model_checkpoint = Path(model_checkpoint)
        assert model_checkpoint.exists()

        results_dir = Path(results_dir) / model_checkpoint.parent.parent.name
        results_dir.mkdir(exist_ok=True, parents=True)

        model = utils.load_model(model_checkpoint).eval()
        state_dict = torch.load(model_checkpoint, map_location='cpu')

        if hasattr(model, 'models'):
            # AverageModel
            res = torch.load(model.ckpts[0], map_location='cpu')
            del res['state_dict']
            del res['optimizer_states']
            state_dict = res

        batch_size = 1 if model.hparams.seq_len > 1000 else state_dict['datamodule_hyper_parameters']['batch_size'] * 2
        if 'enformer' in model.hparams.backbone.lower():
            print('ENFORMER')
            batch_size = 2
            save_batch_size = 2

        datamodule = dataloading.VariantBlockDataModule(
            predict_all_chrom=predict_all_chrom,
            batch_size=batch_size,
            seq_len=model.hparams.seq_len,
            n_sampled_negative=0,
            encode_alt=state_dict['datamodule_hyper_parameters']['encode_alt'],
            predict_all=True,
            seq_order=model.hparams.seq_order,
            # variants_path=state_dict['datamodule_hyper_parameters']['variants_path'],
            variants_path=variants_path,
            use_genostan_labels=model.hparams.use_genostan_labels,
            encode_variant_as=model.hparams.encode_variant_as,
        )

    out_path = results_dir / f'model_predictions_chr{predict_all_chrom}.hdf5'
    if out_path.exists():
        out_path.unlink()

    with torch.no_grad():
        for dataloader in tqdm(
                datamodule.split_predict_dataloaders(n_samples_per_dataloader=save_batch_size),
                total=len(datamodule.dataset_predict) // save_batch_size,
        ):
            trainer = pl.Trainer(
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_progress_bar=False,
                enable_model_summary=False,
            )

            ret = trainer.predict(model=model, dataloaders=dataloader)
            alt = np.concatenate([r[0].cpu().numpy() for r in ret])
            diff = np.concatenate([r[1].cpu().numpy() for r in ret])
            variants = sum([list(r[2]) for r in ret], [])
            rsids = sum([list(r[3]) for r in ret], [])
            chroms = [int(v.split('_')[0].replace('chr', '')) for v in variants]

            with h5py.File(out_path, 'a', libver='latest') as h5_file:
                for chrom, a, d, v, r in zip(chroms, alt, diff, variants, rsids):
                    dset_name = f'{r}-{v}'
                    if dset_name not in h5_file:
                        h5_file.create_dataset(dset_name, data=np.concatenate([a, d]))

                del rsids, chroms, variants, diff, alt, ret


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_checkpoint', type=str, required=True)
    p.add_argument('--variants_path', type=str, default='data/CAUSALdb/credible_set_10k.txt')
    p.add_argument('--results_dir', type=str, default='results/model_predictions/')
    p.add_argument('--save_batch_size', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--load_pretrained_enformer', type=bool, default=False)
    args = p.parse_args()

    generate_predictions(**vars(args))
