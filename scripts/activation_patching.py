from functools import partial
import csv
import einops
import numpy as np
import pandas as pd
import torch
#from circuitsvis.attention import attention_heads
from fancy_einsum import einsum
import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer
import transformer_lens.patching as patching
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from tqdm import tqdm
import os
import argparse
from src.utils import load_triplets, filter_triplets, make_labels, get_avg_logit_diff, prepare_sents
from src.prompts import PROMPTS, models2try

#TODO
"""perform activation patching on given model and data"""

def clean_activation_cache(activation_cache):
    for k in tqdm(activation_cache):
        k2del = activation_cache[k].detach().cpu().numpy()
        del k2del
    del activation_cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 0 means zero change, negative means actively made worse, 1 means totally recovered clean performance, >1 means actively *improved* on clean performance
def preposition_metric(logits:torch.tensor, answer_token_indices: torch.tensor, corrupted_baseline:int, clean_baseline:int):
    return (get_avg_logit_diff(logits, answer_token_indices) - corrupted_baseline) / (clean_baseline  - corrupted_baseline)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type= str, help = "The model to use")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    #instantiate output direcotries
    outdir = f"activation_patching/{args.model}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdir_plots = f"activation_patching/{args.model}/plots"
    if not os.path.exists(outdir_plots):
        os.makedirs(outdir_plots)

    #load model and raw data
    model_name = models2try[args.model]
    model = HookedTransformer.from_pretrained(model_name, cache_dir = "/extra/mattia.proietti/tl_models")
    triplets = load_triplets("./data/triplets_in_on.csv")
    triplets2consider = filter_triplets(triplets, model.tokenizer)
    answ_tokens = make_labels(model.tokenizer, triplets2consider).to(model.cfg.device)
    # print(answ_tokens.shape)
    # print(type(answ_tokens))
    
    for prompt in PROMPTS:
        print(f"Before: GPU memory available at the moment: {torch.cuda.mem_get_info()[0]/1000000000}/ {torch.cuda.mem_get_info()[1]/1000000000}")
        print(f"Patching {prompt}")
        #select the input sentences
        sents_in, sents_on = prepare_sents(PROMPTS[prompt], triplets2consider)
        clean_sents = sents_in + sents_on
        corrupted_sents = sents_on + sents_in
        
        #get corrupted tokens and clean logit
        corrupted_tokens = model.to_tokens(corrupted_sents)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens, device = model.cfg.device)
        CORRUPTED_BASELINE = get_avg_logit_diff(corrupted_logits, answ_tokens).item()
        print(f"Corrupted logitdiff {CORRUPTED_BASELINE}")
        
        #get clean logits and baseline
        clean_logits, clean_cache = model.run_with_cache(clean_sents, device = model.cfg.device)
        CLEAN_BASELINE = get_avg_logit_diff(clean_logits, answ_tokens).item()
        print(f"Clean logitdiff {CLEAN_BASELINE}")
        
        
        prep_metric = partial(preposition_metric,answer_token_indices = answ_tokens, corrupted_baseline = CORRUPTED_BASELINE, clean_baseline = CLEAN_BASELINE)
        #patch results for every block
        every_block_result = patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, prep_metric)

        
        #save output
        file_name = f"patching_{prompt}"
        outpath = os.path.join(outdir, file_name +".pt")
        torch.save(every_block_result.detach().cpu().numpy(), outpath)
        print(f"saved results in {outdir}")

        #try to clean some memory
        for element in [clean_logits, corrupted_logits, every_block_result]:
            element.detach().cpu().numpy()
            del element 
        clean_activation_cache(clean_cache)
        clean_activation_cache(corrupted_cache)
        
        
        gc.collect()
        torch.cuda.empty_cache()
        print(f"After: GPU memory available at the moment: {torch.cuda.mem_get_info()[0]/1000000000}/ {torch.cuda.mem_get_info()[1]/1000000000}")
        
        #TODO make plots

    