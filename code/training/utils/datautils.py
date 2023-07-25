import torch
from torch.utils.data import Dataset, DataLoader

import os
from os.path import join
import subprocess
import logging
import pandas as pd
import pickle as pkl
from itertools import accumulate
import random
from time import time
from utils.train_utils import *


class SMILESProteinDataset(Dataset):
    def __init__(self,  
                 embed_dir,
                 data_path,
                 train:bool,
                 device,
                 gpu,
                 random_state,
                 binary_task:bool,
                 extraction_mode = False):
        start_time = time()
        self.train = train
        self.device = device
        self.gpu = gpu
        self.random_state = random_state
        self.max_prot_seq_len = 1018
        self.max_smiles_seq_len = 256
        self.train_or_test = 'train' if train else 'test'
        self.binary_task = binary_task
        self.embed_dir = embed_dir
        
        self.df = pd.read_csv(join(data_path))
        self.prot_dicts = os.listdir(join(embed_dir, "Protein"))
        self.smiles_dicts = os.listdir(join(embed_dir, "SMILES"))
        self.n_prot_dicts = len(self.prot_dicts)
        self.n_smiles_dicts = len(self.smiles_dicts)
        self.num_subsets = self.n_prot_dicts * self.n_smiles_dicts
        self.total_datacount = len(self.df)
        self.data_counts = []
        self.subset_no = 0
        self.protein_subset_no = -1
        self.smiles_subset_no = 0
        self.update_subset()



    def _load_smiles_repr(self, smiles_repr_file):
        with open(smiles_repr_file, 'rb') as f:
            smiles_rep = pkl.load(f)
        return smiles_rep
    
    def _load_protein_repr(self, protein_repr_path):
        map_loc = self.gpu if is_cuda(self.device) else self.device
        return torch.load(protein_repr_path, map_location=map_loc)
        

    
    def update_subset(self):
        self.protein_subset_no +=1
        self.protein_subset_no  = self.protein_subset_no % self.n_prot_dicts
        self.protein_repr = self._load_protein_repr(join(self. embed_dir, "Protein", self.prot_dicts[self.protein_subset_no]))

        if self.protein_subset_no == 0:
            smiles_repr_file = join(self.embed_dir, "SMILES", self.smiles_dicts[self.smiles_subset_no])
            self.smiles_reprs = self._load_smiles_repr(smiles_repr_file)
            self.smiles_subset_no += 1

        self.subset_no +=1

        all_subset_smiles = list(self.smiles_reprs.keys())
        all_subset_sequences = list(self.protein_repr.keys())

        help_df = self.df.loc[self.df["SMILES"].isin(all_subset_smiles)].copy()
        help_df["index"] = list(help_df.index)
        help_df["Protein sequence"] = [seq[:1018] for seq in help_df["Protein sequence"]]
        help_df = help_df.loc[help_df["Protein sequence"].isin(all_subset_sequences)]

        if self.train:
            help_df = help_df.sample(frac=1, random_state = self.random_state)
        help_df = help_df.reset_index(drop=True)

        #logging.info(f"SMILES subset: {self.smiles_subset_no-1}, Protein Subset: {self.protein_subset_no}, Length help_df: {len(help_df)}")
        self.mappings = help_df.copy()

        #logging.info(self.data_counts)
        if len(self.data_counts) ==0:
            self.data_counts.append(len(help_df))
        else:
            self.data_counts.append(self.data_counts[-1] + len(help_df))


    def __len__(self):
        return self.total_datacount

    def __getitem__(self, idx):
        start_time = time()
        '''This function assumes lienar data reading'''
        prev_subset_max_idx = 0 if self.subset_no == 1 and self.protein_subset_no == 0 else self.data_counts[-2]
        curr_subset_max_idx = self.data_counts[-1]
        idx = idx - prev_subset_max_idx
        #logging.info(f"Item: {idx}, len(help_df): {len(self.mappings)}, prev_subset_max_idx : {prev_subset_max_idx}, curr_subset_max_idx : {curr_subset_max_idx}")
        

        if idx >= len(self.mappings):
            #logging.info(f"updating subset {self.subset_no}")
            self.update_subset()
            while len(self.mappings) == 0:
                self.update_subset()
            prev_subset_max_idx = curr_subset_max_idx
            idx = 0


        label, protein, smiles, index = float(self.mappings["output"][idx]), self.mappings["Protein sequence"][idx], self.mappings["SMILES"][idx], int(self.mappings["index"][idx])

        if self.binary_task:
            label = int(label)

        smiles_emb = self.smiles_reprs[smiles].squeeze()
        protein_emb = torch.from_numpy(self.protein_repr[protein[:1018]])

        smiles_attn_mask = torch.zeros(self.max_smiles_seq_len)
        smiles_attn_mask[:smiles_emb.shape[0]] = 1
        protein_attn_mask = torch.zeros(self.max_prot_seq_len)
        protein_attn_mask[:protein_emb.shape[0]] = 1

        smiles_padding = (0, 0, 0, self.max_smiles_seq_len - smiles_emb.shape[0])
        prot_padding = (0, 0, 0, self.max_prot_seq_len - protein_emb.shape[0])

        smiles_emb = torch.nn.functional.pad(smiles_emb, smiles_padding, mode='constant', value=0)
        protein_emb = torch.nn.functional.pad(protein_emb, prot_padding, mode='constant', value=0)

        labels = torch.Tensor([label])

        labels.requires_grad = False
        smiles_emb = smiles_emb.detach()
        protein_emb = protein_emb.detach()

        return smiles_emb, smiles_attn_mask, protein_emb, protein_attn_mask, labels, index
