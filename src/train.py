import pytorch_lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

import models
import dataloading


def cli_main():
    logger = pytorch_lightning.loggers.WandbLogger(project='MIFM', save_dir='logs')

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.seq_len", "model.seq_len")
            parser.link_arguments("data.seq_order", "model.seq_order")
            parser.link_arguments("data.encode_variant_as", "model.encode_variant_as")

    cli = MyLightningCLI(
        run=False,
        model_class=models.DeepVEP,
        datamodule_class=dataloading.VariantBlockDataModule,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            'logger': logger,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'deterministic': True,
            'log_every_n_steps': 50,
            'callbacks': [
                ModelCheckpoint(monitor='val/loss', mode='min'),
                ModelCheckpoint(filename='last'),
            ],
        }
    )

    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    try:
        from pytorch_lightning.utilities.seed import seed_everything
        seed = cli.config['seed_everything']
        if 'seed_everything' not in logger.experiment.config or logger.experiment.config['seed_everything'] is None:
            logger.experiment.config['seed_everything'] = seed
        seed_everything(seed)
    except ImportError:
        print('Could not import seed everything')

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)


if __name__ == '__main__':
    cli_main()
