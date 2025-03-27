import argparse

import pandas as pd
import torch

from src import utils,dataloading

def main(
        model_checkpoint,
        out_file,
        sequences_file,
        coordinates_file,
        ref_fa_path,
):
    model = utils.load_model(model_checkpoint)

    if sequences_file is not None:
        df_sequences = pd.read_csv(sequences_file, header=None, names=['sequence'])
        dataset = dataloading.SingleVariantDataset(
            df_sequences=df_sequences,
            seq_len=model.hparams.seq_len,
        )
    elif coordinates_file is not None:
        df_variant = pd.read_csv(coordinates_file, sep='\t')
        dataset = dataloading.SingleVariantDataset(
            df_variant=df_variant,
            seq_len=model.hparams.seq_len,
            ref_fa_path=ref_fa_path,
        )

    with torch.no_grad():
        preds = []
        for idx in range(len(dataset)):
            x = dataset[idx]
            pred = model(x)
            preds.append(pred.cpu().numpy()[0])
        pd.DataFrame(preds).to_csv(out_file, index=False, header=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_checkpoint', type=str, default='checkpoints/a7a0q7ar.ckpt')
    p.add_argument('--out_file', type=str)
    p.add_argument('--sequences_file', type=str, default=None)
    p.add_argument('--coordinates_file', type=str, default=None)
    p.add_argument('--ref_fa_path', type=str, default='data/reference/hg19.fa')
    args = p.parse_args()

    main(**vars(args))
