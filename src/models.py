import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics.functional.classification as metrics
import sklearn.metrics as sklearn_metrics
from sklearn.metrics import auc, balanced_accuracy_score



class Transpose(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return torch.transpose(x, *self.axes)


class ConvBlock(nn.Module):
    """
    ConvBlock Layer as defined in Extended Data Fig.1 (Avsec 2021)

    When residual = True, this becomes RConvBlock
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, dropout=0., activation='relu', padding='same',
                 residual=False, bias=False, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        if out_channels is None:
            self.out_channels = in_channels
        else:
            self.out_channels = out_channels

        if residual:
            assert self.out_channels == in_channels, 'Error: when residual = True, the number input channels and output channels must be the same.'
            assert padding == 'same', 'Error: need "same" padding when residual==True'

        self.conv1d_block = nn.Sequential(nn.BatchNorm1d(self.in_channels))

        if activation is not None:
            if activation == 'gelu':
                self.conv1d_block.append(nn.GELU(approximate='tanh'))
            elif activation == 'relu':
                self.conv1d_block.append(nn.ReLU())

        self.conv1d_block.append(
            nn.Conv1d(in_channels, self.out_channels, kernel_size, bias=bias, padding=padding, **kwargs))

        if dropout > 0:
            self.conv1d_block.append(nn.Dropout1d(dropout))

    def forward(self, x):
        if self.residual:
            return x + self.conv1d_block(x)
        else:
            return self.conv1d_block(x)


class Cropping1d(nn.Module):
    """
    Symmetric cropping along position

        crop: number of positions to crop on either side
    """

    def __init__(self, crop, crop_inside=False):
        super().__init__()
        self.crop = crop
        self.crop_inside = crop_inside

    def forward(self, x):
        if self.crop_inside:
            center = x.shape[-1] // 2 - 1
            left_window = self.crop // 2
            return x[:, :, center - left_window:center + self.crop - left_window]
        return x[:, :, self.crop:-self.crop]


class Basenji2ResConvTower(nn.Module):
    """
    Conv Tower as defined in Extended Data Fig.1, green (Avsec 2021)

    The number of channels grows by a constant multiplier from in_channels to out_channels.

    in the original, there is no dropout.
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0., activation='relu', L=6):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.channels_mult = (out_channels / in_channels) ** (1.0 / L)

        self.tower = nn.Sequential(*[
            nn.Sequential(
                ConvBlock(int(self.in_channels * self.channels_mult ** l),
                          int(self.in_channels * self.channels_mult ** (l + 1)), kernel_size, dropout=dropout,
                          activation=activation),
                ConvBlock(int(self.in_channels * self.channels_mult ** (l + 1)),
                          int(self.in_channels * self.channels_mult ** (l + 1)), 1, dropout=dropout,
                          activation=activation, residual=True),
                nn.MaxPool1d(2, 2)
            ) for l in range(L)
        ])

    def forward(self, x):
        return self.tower(x)


class Basenji2DilatedResidualBlock(nn.Module):
    """
    Basenji 2 Dilated Residual Block

    chains together residual dilated convolution blocks, with dilation starting at 2 and growing by a factor of D in ever layer
    """

    def __init__(self, in_channels, kernel_size=3, dropout=0.0, activation='relu', L=4, D=1.5, channel_multiplier=1.):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dropout = dropout
        self.D = D

        self.dilatedresidualblock = nn.Sequential(*[
            ResidualDilatedConvBlock(in_channels=in_channels, kernel_size=3, dropout=dropout, activation=activation,
                                     channel_multiplier=channel_multiplier, dilation=int(2 * D ** l))
            for l in range(L)
        ])

    def forward(self, x):
        return self.dilatedresidualblock(x)


class ResidualDilatedConvBlock(nn.Module):
    """
    Dilated Residual ConvBlock as defined in Extended Data Fig.1, green (Avsec 2021)

    chains together 2 convolutions + dropout, has a residual connection

    the first convolution uses dilated kernels with dilation given by the dilation argument
    the second colvolution is a pointwise convolution

    convolutions have (in_channels * channel_multiplier) and (in_channels) channels, respectively

    in Basenji2, the channel_multiplier is 0.5

    """

    def __init__(self, in_channels, kernel_size=3, dropout=0., dilation=1, activation='relu', channel_multiplier=1.):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.activation = activation
        self.dropout = dropout
        self.dilation = 1
        self.channel_multiplier = channel_multiplier

        self.conv1d_block_1 = ConvBlock(self.in_channels, int(self.out_channels * self.channel_multiplier),
                                        kernel_size=kernel_size, dropout=0, dilation=dilation, activation=activation)
        self.conv1d_block_2 = ConvBlock(int(self.in_channels * self.channel_multiplier), self.out_channels,
                                        kernel_size=1, dropout=dropout, dilation=1, activation=activation)

    def forward(self, x):
        a1 = self.conv1d_block_1(x)
        a2 = self.conv1d_block_2(a1)

        return x + a2


class DeepVEP(pl.LightningModule):
    def __init__(
            self,
            lr=1e-3,
            seq_len=1152,
            seq_order=1,
            n_classes='auto',
            weight_positives=1,
            lr_scheduler=None,
            lr_scheduler_kwargs={},
            encode_variant_as='',
            pooling='max',
            coeff_pool_r=1,

            # aquatic dilated parameters
            C_embed=None,
            use_adaptive_pooling=True,
            crop_inside=True,
            activation='gelu',
            tower_kernel_size=5,
            C=128,
            L1=6,
            L2=2,
            D=1.5,
            crop=4,
            dilated_residual_kernel_size=3,
            dilated_residual_dropout=0.0,
            pointwise_dropout=0.0,
            out_pool_type='max',

            # teacher-student parameters
            teacher_checkpoints=None,
            teacher_sigmoid=False,
            weigh_by_teacher_predictions=None,
            teacher_binary_max=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.h_seq_len = int(seq_len / 2 ** (L1 + 1))
        self.out_seq_len = self.h_seq_len - 2 * crop

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=C // 2,
                kernel_size=15,
                padding='same',
            ),
            ConvBlock(
                in_channels=C // 2,
                out_channels=C // 2,
                kernel_size=1,
                activation=activation,
                residual=True,
            ),
            nn.MaxPool1d(2, 2)
        )
        self.tower = Basenji2ResConvTower(
            C // 2,
            C,
            kernel_size=tower_kernel_size,
            L=L1,
            activation=activation,
        )
        self.dilated_residual_block = Basenji2DilatedResidualBlock(
            self.tower.tower[-1][-2].out_channels,
            kernel_size=dilated_residual_kernel_size,
            dropout=dilated_residual_dropout,
            D=D,
            channel_multiplier=1.,
            L=L2,
            activation=activation,
        )
        self.cropping = Cropping1d(crop, crop_inside=crop_inside)
        self.pointwise = ConvBlock(
            self.dilated_residual_block.dilatedresidualblock[-1].out_channels,
            2 * C,
            kernel_size=1,
            dropout=pointwise_dropout,
            activation=activation,
        )
        self.features = nn.Sequential(
            self.stem,
            self.tower,
            self.dilated_residual_block,
            self.cropping,
            self.pointwise,
        )
        self.pooling = nn.AdaptiveAvgPool1d(1) if use_adaptive_pooling else nn.Identity()
        out_seq_len = 1 if use_adaptive_pooling else crop

        self.head = nn.Sequential(nn.Linear(2 * C * out_seq_len, 1))
        self.model = nn.Sequential(
            self.features,
            self.pooling,
            nn.Flatten(),
            self.head,
        )

    @property
    def num_tokens(self):
        return 5 if self.hparams.encode_variant_as == 'V' else 4

    @property
    def in_channels(self):
        return self.num_tokens ** self.hparams.seq_order

    def forward(self, x):
        return self.model(x)

    @property
    def is_student_model(self):
        return self.hparams.teacher_checkpoints is not None

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            lr=self.hparams.lr,
            params=self.parameters(),
        )
        if self.hparams.lr_scheduler is not None:
            sched_cls = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler)
            scheduler = sched_cls(optimizer=optim, **self.hparams.lr_scheduler_kwargs)
            return [optim], [scheduler]

        return optim

    def training_step(self, batch, batch_idx=None):
        return self._step(batch=batch, batch_idx=batch_idx, step_name='train')

    def loss_fn(self, y, y_hat, sample_weights=None):
        if self.is_student_model:
            if sample_weights is not None:
                return (sample_weights * ((y - y_hat) ** 2)).mean()
            return F.mse_loss(input=y_hat, target=y)

        if self.hparams.pooling in (
                'generalized_mean',
                'isr',
        ):
            return F.binary_cross_entropy(input=y_hat, target=y, weight=sample_weights)
        return F.binary_cross_entropy_with_logits(input=y_hat, target=y, weight=sample_weights)

    def _step(self, batch, step_name, batch_idx=None):
        def log(val, name):
            if step_name != 'predict':
                self.log(
                    f'{step_name}/{name}',
                    val,
                    prog_bar=True,
                    batch_size=y_hat.shape[0],
                    logger=True,
                    on_step='train' in step_name,
                    on_epoch='train' not in step_name,
                )

        x, y, block_lengths, block_infos = batch
        num_pos = sum(l for l in block_lengths[:len(block_infos)])

        preds = self.forward(x)

        if self.is_student_model:
            with torch.no_grad():
                x_teacher = x

                if self.teacher_models[
                    0].hparams.encode_variant_as == 'V' and self.hparams.encode_variant_as != 'V':
                    x_teacher = torch.cat([
                        x_teacher,
                        torch.zeros_like(x_teacher)[:, :1],
                    ], dim=1)
                    x_teacher[:, :4, x.shape[-1] // 2 - 1] = 0
                    x_teacher[:, 4, x.shape[-1] // 2 - 1] = 1

                ys = torch.stack([teacher.forward(x_teacher) for teacher in self.teacher_models], dim=0)
                if self.hparams.teacher_sigmoid:
                    ys = F.sigmoid(ys)
                y = ys.mean(dim=0)

                if self.hparams.teacher_binary_max:
                    y[num_pos:] = 0

                    i = 0
                    for block_length in block_lengths[:len(block_infos)]:
                        block = y[i:i + block_length]
                        _, idx_max = block.max(dim=0)
                        y[i:i + block_length] = 0
                        y[i + idx_max] = 1
                        i += block_length

            y_hat = preds
            y_sigmoid = y if self.hparams.teacher_sigmoid else F.sigmoid(y)
            y_hat_sigmoid = F.sigmoid(y_hat)
            y_hat_binary = y_hat_sigmoid > .5
            n_causal = y_hat_binary[:num_pos].sum().float() / len(block_infos)
            if self.hparams.teacher_sigmoid:
                y_hat = y_hat_sigmoid

            sample_weights = None
            if self.hparams.weigh_by_teacher_predictions is not None:
                # sample_weights = self.hparams.weigh_by_teacher_predictions * y_sigmoid
                sample_weights = (self.hparams.weigh_by_teacher_predictions - 1) * (y_sigmoid > .5) + 1
            loss = self.loss_fn(
                y_hat=y_hat,
                y=y,
                sample_weights=sample_weights,
            )

            logs = {
                'loss': loss,
                'n_causal': n_causal,
                'mean_pred': y_hat_sigmoid.mean(),
                'max_pred': y_hat_sigmoid.max(),
                'max_y': y_sigmoid.max(),
            }
            for k, v in logs.items():
                log(v, k)

            return logs

        y_hat = []
        block_start = 0

        for block_idx, block_len in enumerate(block_lengths):
            preds_block = preds[block_start:block_start + block_len]
            r = self.hparams.coeff_pool_r
            if self.hparams.pooling == 'max':
                block_pred = preds_block.max(dim=0)[0]
            elif self.hparams.pooling == 'log_sum_exp':
                block_pred = (r * preds_block).exp().mean(dim=0).log() / r
            elif self.hparams.pooling == 'generalized_mean':
                preds_block = F.sigmoid(preds_block)
                if block_len > 1:
                    block_pred = preds_block.pow(r).mean(dim=0).pow(1 / r)
                else:
                    block_pred = preds_block[0]
            elif self.hparams.pooling == 'isr':
                preds_block = F.sigmoid(preds_block)
                if block_len > 1:
                    preds_block_hat = (preds_block / (1 - preds_block)).sum(dim=0)
                    block_pred = preds_block_hat / (1 + preds_block_hat)
                else:
                    block_pred = preds_block[0]
            y_hat.append(block_pred)

            block_start = block_start + block_len

        y_hat = torch.stack(y_hat)
        with torch.no_grad():
            y = torch.zeros_like(y_hat)
            y[:len(block_infos)] = 1

        y_hat_pos, y_pos = y_hat[:len(block_infos)], y[:len(block_infos)]
        loss_pred_pos = self.loss_fn(y_pos, y_hat_pos)
        y_hat_neg, y_neg = y_hat[len(block_infos):], y[len(block_infos):]

        neg_sample_weights = None
        loss_pred_neg = self.loss_fn(y_neg, y_hat_neg, sample_weights=neg_sample_weights)

        loss = loss_pred_pos * self.hparams.weight_positives + loss_pred_neg


        log(loss, 'loss')
        log(y_hat.mean(), 'preds_mean')
        log(y_hat.std(), 'preds_std')
        log(loss_pred_pos, 'loss_pred_pos')
        log(loss_pred_neg, 'loss_pred_neg')

        with torch.no_grad():
            if not self.hparams.pooling in ('generalized_mean', 'isr'):
                y_hat_pos = F.sigmoid(y_hat_pos)
                y_hat = F.sigmoid(y_hat)
                y_hat_neg = F.sigmoid(y_hat_neg)
            log(metrics.binary_accuracy(preds=y_hat_pos, target=y_pos), 'acc_pos')
            log(metrics.binary_accuracy(preds=y_hat_neg, target=y_neg), 'acc_neg')
            log(metrics.binary_f1_score(preds=y_hat, target=y), 'f1')
            log(metrics.binary_accuracy(preds=y_hat, target=y), 'acc')
            log(metrics.binary_f1_score(preds=y_hat_pos, target=y_pos), 'f1_pos')
            log(metrics.binary_recall(preds=y_hat_pos, target=y_pos), 'recall_pos')
            log(metrics.binary_recall(preds=y_hat, target=y), 'recall')
            log(metrics.binary_precision(preds=y_hat_pos, target=y_pos), 'precision_pos')
            log(metrics.binary_precision(preds=y_hat, target=y), 'precision')

            if step_name == 'val':
                log(sklearn_metrics.roc_auc_score(y.cpu().numpy(), y_hat.cpu().numpy()), 'auroc')
                p, r, _ = sklearn_metrics.precision_recall_curve(
                    y.cpu().numpy(), y_hat.cpu().numpy())
                log(auc(r, p), 'auprc')


        if torch.isnan(loss) or torch.isinf(loss):
            return None

        return loss