import sys
import json
import copy
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data.distributed import DistributedSampler
from .train_utils import *
from transformers import BertConfig



class MM_TNConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
        self,
        s_hidden_size=767,
        p_hidden_size=1280,
        max_seq_len=1276,

    ):
        self.s_hidden_size = s_hidden_size
        self.p_hidden_size = p_hidden_size
        self.max_seq_len = max_seq_len
        self.binary_task = binary_task

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config



class BertSmilesPooler(nn.Module):
    def __init__(self, config):
        super(BertSmilesPooler, self).__init__()
        self.dense = nn.Linear(config.s_hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class BertProteinPooler(nn.Module):
    def __init__(self, config):
        super(BertProteinPooler, self).__init__()
        self.dense = nn.Linear(config.p_hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class Bert(nn.Module):
    def __init__(self, num_classes, emb_dim, no_layers, binary_task):
        super(Bert, self).__init__()
        self.config = BertConfig(
            hidden_size=emb_dim,
            num_hidden_layers=no_layers,
            num_attention_heads=6,
        )
        self.binary_task = binary_task
        self.num_classes = num_classes
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_hidden_layers= self.config.num_hidden_layers

        # transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    dim_feedforward=4 * self.config.hidden_size,
                    activation="gelu",
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        )

        # output layer
        self.hidden_layer = nn.Linear(self.config.hidden_size, 32)
        self.output_layer = nn.Linear(32, num_classes)
        self.sigmoid_layer = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, input_ids, attention_mask, get_repr=False):
        x = input_ids.permute(1, 0, 2)
        # transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, src_key_padding_mask=attention_mask)
        
        if get_repr:
            hidden_repr = x[0,:,:]
            
        x = x[0,:,:]
        x = x.reshape(-1, self.hidden_size)
        x = self.hidden_layer(x)

        x = self.ReLU(x)
        x = self.output_layer(x)

        if self.binary_task:
            x = self.sigmoid_layer(x)
        
        if get_repr:
            return x, hidden_repr
        return x


class MM_TN(nn.Module):
    def __init__(self, config):
        super(MM_TN, self).__init__()

        self.config = config
        self.s_pooler = BertSmilesPooler(config)
        self.p_pooler = BertProteinPooler(config)
        self.main_bert = Bert(num_classes=1,  emb_dim=config.hidden_size,no_layers=config.num_hidden_layers, binary_task = config.binary_task)
        
    def extract_repr(self):
        pass

    def forward(
        self,
        smiles_emb,
        smiles_attn,
        protein_emb,
        protein_attn,
        device,
        gpu,
        get_repr=False
    ):
        batch_size, _, _ = smiles_emb.shape
        s_embedding = self.s_pooler(smiles_emb)
        p_embedding = self.p_pooler(protein_emb)
        
        
        zeros_pad = torch.zeros((batch_size, 1, self.config.hidden_size)).to(device)
        zeros_mask = torch.zeros(batch_size,1).to(device)
        ones_pad = torch.ones((batch_size, 1, self.config.hidden_size)).to(device)
        ones_mask = torch.ones(batch_size,1).to(device)
        
        if is_cuda(device):
            zeros_pad, zeros_mask, ones_pad, ones_mask = zeros_pad.cuda(gpu), zeros_mask.cuda(gpu), ones_pad.cuda(gpu), ones_mask.cuda(gpu)
        
        concat_seq = torch.cat((ones_pad, s_embedding, zeros_pad, p_embedding), dim=1) # <cls> SMILES <sep> Protein
        attention_mask = torch.cat((ones_mask, smiles_attn, zeros_mask, protein_attn), dim=1)

        if get_repr:
            output, final_repr = self.main_bert(concat_seq, attention_mask, get_repr)
            return output, final_repr
            
        else:
            output = self.main_bert(concat_seq, attention_mask)
            return output

        

