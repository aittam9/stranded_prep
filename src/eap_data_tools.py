from functools import partial

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 

from typing import Literal, List, Dict, Union, Tuple, Optional
from src.utils import prepare_sents, make_labels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#adapted from micheal hanna tutorial/github
def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, df):
        #self.df = pd.read_csv(filepath)
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    logits = get_logit_positions(logits, input_length)
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results

####--------------------------------------------------------------------------------------------------------------####
#brand new code
#load model configured as needed by the library for eap analysis
def load_model_eap(model_name:str):
    model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=DEVICE,
        dtype=torch.float16,
        cache_dir = "/extra/mattia.proietti/tl_models"
    )
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    return model

#helper to format data for eap
def make_eap_input_df(prompt, triplets2consider, model):
    """
    Format a df as expected by a EAPDataset and dataloader
    """
    # format and prepare the sents
    sents_in, sents_on = prepare_sents(prompt, triplets2consider)
    clean = sents_in + sents_on 
    corrupted = sents_on + sents_in 
    #make labels index lists
    labels = make_labels(model.tokenizer, triplets2consider) 
    correct_idx, incorrect_idx = labels[:,0].tolist(), labels[:,1].tolist()
    #make the dataframe
    df = pd.DataFrame(zip(clean, corrupted, correct_idx, incorrect_idx), columns = ["clean", "corrupted", "correct_idx", "incorrect_idx"])
    return df