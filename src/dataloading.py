import os
import math

import pandas as pd
import torch
import torch.utils.data as torch_data
import pytorch_lightning as pl

from torch import permute, tensor
from torch.nn import Embedding
from numpy.typing import ArrayLike
from typing import Union, Optional

from pysam import FastaFile
import numpy as np
from itertools import product
import torch

_NUM_CPUS_AVAILABLE = 1
# try to infer the number of available CPUs on an HPC cluster
NUM_WORKERS_ENV = int(os.environ.get('NSLOTS', os.environ.get('SLURM_JOB_CPUS_PER_NODE', _NUM_CPUS_AVAILABLE)))
print(f'Num workers: {NUM_WORKERS_ENV}')


class DNATokenizer:
    """
    Class that handles breaking up DNA-sequence-strings into oligo-nucleotide sequences and mapping to integer token IDs
    """

    def __init__(
            self,
            seq_order: int = 1,
            stride: Optional[int] = None,
            allow_N: bool = True,
            N_max: int = 0,
    ):

        if allow_N:
            # maps (oligo-) nucleotides to integer token-IDs, e.g., AAA -> 0
            self.mapping = {''.join(x): i for i, x in
                            enumerate(list(product(*[['A', 'C', 'G', 'N', 'T'] for _ in range(seq_order)])))}
            # maps (oligo-) nucleotides to the integer token-IDs of their reverse-complement, e.g., AAA -> 124 (= TTT in the original mapping), or ACG -> 39 (= CGT in the original mapping)
            self.mapping_rc = {''.join(x[::-1]): i for i, x in
                               enumerate(list(product(*[['T', 'G', 'C', 'N', 'A'] for _ in range(seq_order)])))}
        else:
            # simplified version that doesn't have "N" (unknown) nucleotides
            self.mapping = {''.join(x): i for i, x in
                            enumerate(list(product(*[['A', 'C', 'G', 'T'] for _ in range(seq_order)])))}
            self.mapping_rc = {''.join(x[::-1]): i for i, x in
                               enumerate(list(product(*[['T', 'G', 'C', 'A'] for _ in range(seq_order)])))}

        self.stride = stride if stride is not None else 1
        self.seq_order = seq_order
        self.allow_N = allow_N
        self.N_max = N_max

        # sanity check
        # self._validate_rc_mapping()

    def _validate_rc_mapping(self):
        # validate the mapping is working as expected...
        rc_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        for k, v in self.mapping_rc.items():
            rc = k[::-1]
            rc = ''.join(rc_dict[o] for o in rc)
            print('{} -> {}, {} -> {}'.format(rc, self.mapping[rc], k, self.mapping_rc[k]))
            assert self.mapping[rc] == self.mapping_rc[
                k], 'Error: invalid reverse-complement mapping. please report this bug'

    def get_one_hot_weights_matrix(self, N_max: int = 0, reverse_complement=False):

        '''
        Returns a weight matrix which corresponds to 1-hot encoding of (oligo-)nucleotides, and the corresponding positional mapping (dict)

        The weight matrix can be passed to torch.nn.Embedding

        N_max controls the max number of unknown nucleotides ("N" in the FASTA) allowed before sequences are mapped to the null-encoding (vector of zeros).
        The default (N_max = 0) excludes (oligo-) nucleotides with any Ns (-> blinding)
        '''

        if not self.allow_N:
            idx = self.mapping if not reverse_complement else self.mapping_rc
            if N_max > 0:
                raise ValueError('N_max > 0, but object was created with allow_N = False !')
        else:
            mapping = self.mapping if not reverse_complement else self.mapping_rc
            idx = {}
            i = 0
            for k in mapping.keys():
                if str(k).count('N') > N_max:
                    continue
                else:
                    idx[k] = i
                    i += 1

        dim_embed = max(idx.values()) + 1

        weights = np.zeros((len(self.mapping), dim_embed))

        for i, (k, v) in enumerate(self.mapping.items()):
            if k in idx.keys():
                weights[i, idx[k]] = 1.

        return weights, idx

    def tokenize(self, sequences: Union[str, ArrayLike], reverse_complement: bool = False, random: bool = False):

        """
        Turns an array of DNA-strings (n,) into an array of (oligo-)nucleotide tokens
        If reverse_complement == True, will return the reverse-complement mapping
        """

        if random:
            reverse_complement = np.random.choice([0, 1]).astype(bool)

        if reverse_complement:
            if isinstance(sequences, str):
                return np.array([self.mapping_rc[sequences[i * self.stride:(i * self.stride + self.seq_order)]] for i in
                                 range((len(sequences) - self.seq_order) // self.stride + 1)])[np.newaxis, ::-1].copy()
            else:
                return np.array([[self.mapping_rc[s[i * self.stride:(i * self.stride + self.seq_order)]] for i in
                                  range((len(s) - self.seq_order) // self.stride + 1)] for s in sequences])[:,
                       ::-1].copy()

        else:
            if isinstance(sequences, str):
                return np.array([self.mapping[sequences[i * self.stride:(i * self.stride + self.seq_order)]] for i in
                                 range((len(sequences) - self.seq_order) // self.stride + 1)])[np.newaxis, :]
            else:
                return np.array([[self.mapping[s[i * self.stride:(i * self.stride + self.seq_order)]] for i in
                                  range((len(s) - self.seq_order) // self.stride + 1)] for s in sequences])

    def onehot_reverse_complement_func(self, use_numpy=False):

        """
        returns a function that peforms reverse complement mapping after one-hot encoding
        useful for things like variant effect prediction
        """

        # these return different mapping from those stored in self.mapping and self.rc_mapping, because "N" nucleotides can be dropped.
        _, mapping = self.get_one_hot_weights_matrix(N_max=self.N_max, reverse_complement=False)
        _, mapping_rc = self.get_one_hot_weights_matrix(N_max=self.N_max, reverse_complement=True)

        rc_permute = [mapping_rc[k] for k, _ in mapping.items()]

        if use_numpy:
            def rcfun(x: np.ndarray):
                # this returns a view
                return x[:, rc_permute, ::-1]

            return rcfun
        else:
            def rcfun(x: torch.Tensor):
                # this copies the data
                return torch.flip(x[:, rc_permute, :], (2,))

            return rcfun

class BlockDataset(torch_data.Dataset):
    def __init__(
            self,
            ref_fa_path,
            df_variant,
            seq_len,
            df_variant_neg=None,
            max_block_size=127,
            use_secondary_signals=False,
            dtype=torch.float32,
            seq_order=1,
            stride=1,
            tie="l",
            encode_variant_as='',
            keep_original_V_nucletotide=False,
            add_no_variant_negative=False,
            n_sampled_negative=1,
            random_shift=0,
            random_reverse_complement=False,
    ):
        self.keep_original_V_nucletotide = keep_original_V_nucletotide
        self.encode_variant_as = encode_variant_as

        self.add_no_variant_negative = add_no_variant_negative
        self.random_reverse_complement = random_reverse_complement
        self.random_shift = random_shift
        self.use_secondary_signals = use_secondary_signals
        self.n_sampled_negative = max_block_size if n_sampled_negative is None else n_sampled_negative
        self.dtype = dtype
        self.seq_order = seq_order
        self.ref_fa_path = ref_fa_path
        self.ref = FastaFile(ref_fa_path)
        assert os.path.isfile(ref_fa_path + '.fai'), 'Error: no index found for Fasta-file: {}'.format(ref_fa_path)
        self.seq_len = seq_len
        assert tie in ['l', 'r']
        self.tie = tie
        if (self.seq_len % 2) == 0:
            self._left_window = int(self.seq_len / 2 - 1) if self.tie == 'l' else int(self.seq_len / 2)
            self._right_window = int(self.seq_len / 2 + 1) if self.tie == 'l' else int(self.seq_len / 2)
        else:
            self._left_window = math.floor(self.seq_len / 2)
            self._right_window = math.ceil(self.seq_len / 2)

        self.tokenizer = DNATokenizer(seq_order=seq_order, allow_N=True, stride=stride)
        W, mapping = self.tokenizer.get_one_hot_weights_matrix(N_max=0)
        dna_embed = Embedding.from_pretrained(tensor(W), freeze=True)
        self.transform = lambda x: permute(dna_embed(tensor(x)), [0, 2, 1])

        self.df_variant = df_variant
        self.df_variant_neg = df_variant_neg

        if 'block_id' in self.df_variant:
            self.block_variants = []
            self.block_labels = []
            self.block_info = []
            cols = ['chr', 'bp', 'rsid', 'ea', 'nea',
                    # 'variant'
                    ]

            # initialize variants for each block and max labels for all GWAS studies within a block
            for _, df_block in self.df_variant.groupby(
                    ['block_id', 'primary_mode'] if self.use_secondary_signals else 'block_id'
            ):
                if max_block_size is None:
                    block = df_block.drop_duplicates('rsid')[cols]
                else:
                    block = df_block.sort_values('p').drop_duplicates('rsid').iloc[:max_block_size][cols]
                self.block_variants.append(block)

                self.block_info.append({
                    # 'variant': block['variant'],
                    'rsid': block['rsid'],
                })

    def __len__(self):
        return len(self.block_variants)

    def __getitem__(self, idx):
        ref, alt = self.get_block_sequences(idx)

        block_info = self.block_info[idx]

        pos = ref

        if self.n_sampled_negative > 0:
            neg = self.sample_negative_sequences(n=self.n_sampled_negative)
        if self.df_variant_neg is None:
            neg = alt

        if self.add_no_variant_negative and self.encode_variant_as == 'V':
            pos_no_V = pos.clone()
            pos_no_V[:, -1] = 0

            neg_no_V = neg.clone()
            neg_no_V[:, -1] = 0

            # clean up the true nucleotide from neg
            idxs_var = torch.where(neg[:, 4])
            neg[idxs_var[0], :4, idxs_var[1]] = 0

            neg = torch.cat([neg, neg_no_V, pos_no_V], dim=0)

            # clean up the true nucleotide from pos
            idxs_var = torch.where(pos[:, 4])
            pos[idxs_var[0], :4, idxs_var[1]] = 0

        label_neg = torch.zeros((neg.shape[0], 1))
        label_pos = torch.ones((pos.shape[0], 1))

        return {
            'pos': pos,
            'neg': neg,
            'label_pos': label_pos,
            'label_neg': label_neg,
            'block_info': block_info,
        }

    @property
    def num_tokens(self):
        return 5 if self.encode_variant_as == 'V' else 4

    def get_variant_sequence_str(self, variant):
        """get reference sequence for a variant"""
        pos = variant.bp - 1

        full_seq = self.ref.fetch(f'chr{variant.chr}', pos - self.left_window, pos + self.right_window).upper()
        if '>' in full_seq:
            idx_last = full_seq.index('>')
            full_seq = full_seq[:idx_last] + 'N' * (len(full_seq) - idx_last)

        full_seq = full_seq.replace('>', 'N')
        full_seq_ref = full_seq
        if self.encode_variant_as is not None:
            variant['ea'] = variant['nea']

        len_ref = len(variant['nea'])
        len_alt = len(variant['ea'])
        d = len_alt - len_ref if len_alt < len_ref else 0
        if d < 0:
            # ref longer than alt
            full_seq = self.ref.fetch(f'chr{variant.chr}', pos - self.left_window, pos + self.right_window - d).upper()
            if '>' in full_seq:
                idx_last = full_seq.index('>')
                full_seq = full_seq[:idx_last] + 'N' * (len(full_seq) - idx_last)

        left_seq = full_seq[:self.left_window]
        right_seq_alt = full_seq[(self.left_window + len_ref):][:(self.right_window - len_alt - d)]
        full_seq_alt = f'{left_seq}{variant["ea"]}{right_seq_alt}'.upper()

        # if self.random_shift != 0:
        #     shift = np.random.randint(-self.random_shift, self.random_shift + 1)
        #     full_seq_ref = full_seq_ref[self.random_shift + shift:(-self.random_shift + shift) or None]
        #     full_seq_alt = full_seq_alt[self.random_shift + shift:(-self.random_shift + shift) or None]

        return full_seq_ref, full_seq_alt

    @property
    def left_window(self):
        return self._left_window + self.random_shift

    @property
    def right_window(self):
        return self._right_window + self.random_shift

    def sample_negative_sequences(self, n):
        seqs = []
        for _, variant in self.df_variant_neg.sample(n=min(n, len(self.df_variant_neg))).iterrows():
            seqs.extend(self.get_variant_sequences(variant))
        seqs = torch.cat(seqs, dim=0)

        return seqs

    def get_variant_sequences(self, variant):
        try:
            ref, alt = self.get_variant_sequence_str(variant)

            ref = self.tokenizer.tokenize(ref, random=self.random_reverse_complement)
            alt = self.tokenizer.tokenize(alt, random=self.random_reverse_complement)
            ref, alt = self.transform(ref), self.transform(alt)

            if self.encode_variant_as is not None and len(self.encode_variant_as) > 0:
                ref = self.encode_variant(ref)
                alt = self.encode_variant(alt)

            if self.random_shift != 0:
                shift = np.random.randint(-self.random_shift, self.random_shift + 1)
                ref = ref[:, :, self.random_shift + shift:(-self.random_shift + shift) or None]
                alt = alt[:, :, self.random_shift + shift:(-self.random_shift + shift) or None]

            if self.seq_order > 1:
                if ref.shape[-1] < self.seq_len:
                    ref = torch.cat(
                        [ref, torch.zeros(1, self.num_tokens ** self.seq_order, self.seq_len - ref.shape[-1])],
                        dim=-1,
                    )
                if alt.shape[-1] < self.seq_len:
                    alt = torch.cat(
                        [alt, torch.zeros(1, self.num_tokens ** self.seq_order, self.seq_len - alt.shape[-1])],
                        dim=-1,
                    )

            if ref.shape[-1] > self.seq_len or alt.shape[-1] > self.seq_len:
                print(f"too long of a sequence: {ref.shape[-1]}")
                ref, alt = ref[:, :, :self.seq_len], alt[:, :, :self.seq_len]
            if ref.shape[-1] < self.seq_len:
                print('ref too short')
                ref = torch.cat([ref, torch.zeros(1, self.num_tokens, self.seq_len - ref.shape[-1])], dim=-1)
            if alt.shape[-1] < self.seq_len:
                print('alt too short')
                alt = torch.cat([alt, torch.zeros(1, self.num_tokens, self.seq_len - alt.shape[-1])], dim=-1)

        except Exception as e:
            print(e)
            print(variant)
            ref = torch.zeros((1, self.num_tokens ** self.seq_order, self.seq_len))
            alt = torch.zeros((1, self.num_tokens ** self.seq_order, self.seq_len))

        return ref.to(self.dtype), alt.to(self.dtype)

    def get_block_sequences(self, idx):
        refs, alts = [], []

        for _, variant in self.block_variants[idx].iterrows():
            ref, alt = self.get_variant_sequences(variant)
            refs.append(ref)
            alts.append(alt)

        return torch.cat(refs, dim=0), torch.cat(alts, dim=0)

def block_collate_fn(batch):
    # input sequences
    inputs = torch.cat([b['pos'] for b in batch] + [b['neg'] for b in batch])

    return (
        inputs,
        # labels
        torch.cat([b['label_pos'] for b in batch] + [b['label_neg'] for b in batch]),
        # block lengths
        [b['pos'].shape[0] for b in batch] + [1] * sum(b['neg'].shape[0] for b in batch),
        # info for positive blocks
        [b['block_info'] for b in batch],
    )

class VariantBlockDataModule(pl.LightningDataModule):
    def __init__(
            self,
            variants_path,
            negative_variants_path,
            ref_fa_path='data/reference/hg19.fa',
            seq_len=512,
            batch_size=4,
            seq_order=1,
            encode_variant_as='',
            min_variants_per_block=1,
            use_secondary_signals=False,
            n_sampled_negative=1,
            random_shift=0,
            random_reverse_complement=False,
            max_block_size=127,
            keep_original_V_nucletotide=False,
            val_chromosomes=('11', '12'),
            test_chromosomes=('9', '10'),
    ):
        super().__init__()

        variants = pd.read_csv(
            variants_path,
            sep='\t',
        )
        variants['ea'] = variants['ea'].fillna('')
        if min_variants_per_block > 1:
            variants = pd.concat([
                group for _, group in variants.groupby('block_id') if group['rsid'].nunique() >= min_variants_per_block
            ])

        variants_neg = pd.read_csv(
            negative_variants_path,
            sep='\t',
        )

        val_idxs = variants['chr'].isin(val_chromosomes)
        test_idxs = variants['chr'].isin(test_chromosomes)
        variants_train = variants.loc[(~test_idxs) & (~val_idxs)]
        variants_val = variants.loc[val_idxs]

        val_idxs = variants_neg['chr'].isin(val_chromosomes)
        test_idxs = variants_neg['chr'].isin(test_chromosomes)
        variants_neg_train = variants_neg.loc[(~test_idxs) & (~val_idxs)]
        variants_neg_val = variants_neg.loc[val_idxs]

        self.dataset_train = BlockDataset(
            ref_fa_path=ref_fa_path,
            df_variant=variants_train,
            df_variant_neg=variants_neg_train,
            seq_len=seq_len,
            use_secondary_signals=use_secondary_signals,
            n_sampled_negative=n_sampled_negative,
            random_shift=random_shift,
            random_reverse_complement=random_reverse_complement,
            seq_order=seq_order,
            max_block_size=max_block_size,
            encode_variant_as=encode_variant_as,
            keep_original_V_nucletotide=keep_original_V_nucletotide,
        )

        self.save_hyperparameters()

    def train_dataloader(self):
        return torch_data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=block_collate_fn,
            num_workers=NUM_WORKERS_ENV,
        )

class SingleVariantDataset(torch_data.Dataset):
    """For downstream predictions"""
    def __init__(
            self,
            seq_len=None,
            df_sequences=None,
            df_variant=None,
            ref_fa_path=None,
            seq_order=1,
            stride=1,
            tie='l',
            dtype=torch.float32,
    ):
        self.df_variant = df_variant
        self.df_sequences = df_sequences

        if self.df_variant is not None:
            self.ref_fa_path = ref_fa_path
            self.ref = FastaFile(ref_fa_path)
            assert os.path.isfile(ref_fa_path + '.fai'), 'Error: no index found for Fasta-file: {}'.format(ref_fa_path)
            self.seq_len = seq_len
            assert tie in ['l', 'r']
            self.tie = tie
            if (self.seq_len % 2) == 0:
                self.left_window = int(self.seq_len / 2 - 1) if self.tie == 'l' else int(self.seq_len / 2)
                self.right_window = int(self.seq_len / 2 + 1) if self.tie == 'l' else int(self.seq_len / 2)
            else:
                self.left_window = math.floor(self.seq_len / 2)
                self.right_window = math.ceil(self.seq_len / 2)
        elif self.df_sequences is None:
            raise ValueError('df_variant or df_sequences must be provided')

        self.dtype = dtype
        self.seq_order = seq_order
        self.tokenizer = DNATokenizer(seq_order=seq_order, allow_N=True, stride=stride)
        W, mapping = self.tokenizer.get_one_hot_weights_matrix(N_max=0)
        dna_embed = Embedding.from_pretrained(tensor(W), freeze=True)
        self.transform = lambda x: permute(dna_embed(tensor(x)), [0, 2, 1])


    def __len__(self):
        return len(self.df_variant or self.df_sequences)

    def __getitem__(self, idx):
        if self.df_variant is not None:
            seq = self.get_variant_sequence_str(self.df_variant.iloc[idx])
        else:
            seq = self.df_sequences.iloc[idx]
        seq = self.tokenizer.tokenize(seq)
        seq = self.transform(seq)

        return seq.to(self.dtype)

    def get_variant_sequence_str(self, variant):
        """get reference sequence for a variant"""
        pos = variant.bp - 1

        full_seq = self.ref.fetch(f'chr{variant.chr}', pos - self.left_window, pos + self.right_window).upper()
        if '>' in full_seq:
            idx_last = full_seq.index('>')
            full_seq = full_seq[:idx_last] + 'N' * (len(full_seq) - idx_last)

        full_seq = full_seq.replace('>', 'N')

        return full_seq