import csv
import einops
import numpy as np
import pandas as pd
import os
import torch
from fancy_einsum import einsum
from transformer_lens import  HookedTransformer
import matplotlib.pyplot as plt
import gc
from utils import load_triplets, make_labels, filter_triplets, get_avg_logit_diff, prepare_sents, residual_stack_to_logit_diff
from argparse import ArgumentParser
from prompts import PROMPTS, models2try
import seaborn as sns


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_PATH = "./triplets_in_on.csv"

if __name__ == "__main__":
    #TODO make a single script with possibility to choose model, and logit-lens type (res-stream or layers)
    parser = ArgumentParser()
    parser.add_argument("--model", "-m",type=str, help = "The hugginface model id we want to use")
    parser.add_argument("--type", "-t", type=str, help = "The type of logit-lens we want to perform. Choose between residual and layers")
    args = parser.parse_args()

    if not args.model in models2try.keys():
         print(f"{args.model} not in the list of the available models.Please use one of the following keys:")
         print(list(models2try.keys()))
         exit 
          
    
    output_dir = f"./exploratory_analysis/{args.model}"
    if not os.path.exists(f"./exploratory_analysis"):
         os.mkdir(f"./exploratory_analysis")
    if not os.path.exists(output_dir):
         os.mkdir(output_dir)
    
    print(os.getcwd())
    print(f"\n\nUsing: {DEVICE}")
    print(f"GPU memory available at the moment: {torch.cuda.mem_get_info()[0]/1000000000}/ {torch.cuda.mem_get_info()[1]/1000000000}")
    print(f"Before: GPU memory available at the moment: {torch.cuda.mem_get_info()[0]/1000000000}/ {torch.cuda.mem_get_info()[1]/1000000000}")
    
    #load model
    model_id = models2try[args.model]
    model = HookedTransformer.from_pretrained(model_id, cache_dir = "/extra/mattia.proietti/tl_models")
    
    #load and filter triplets
    triplets = load_triplets(INPUT_PATH)
    triplets2consider = filter_triplets(triplets, model.tokenizer)
    
    #init needed variables
    store_caches = {}
    answ_tokens = make_labels(model.tokenizer, triplets2consider).to(DEVICE)
    answer_residual_directions = model.tokens_to_residual_directions(answ_tokens)
    logit_diff_directions = (answer_residual_directions[:, 0] - answer_residual_directions[:, 1])
      
    method = args.type
    logit_lens_dict = {}
    for prompt in PROMPTS:
        gc.collect()
        torch.cuda.empty_cache()
        #torch.cuda.ipc_collect()
        store_caches[prompt] = None
        
        #format the template removing triplets containing multi-tokens
        sents_in, sents_on = prepare_sents(PROMPTS[prompt], triplets2consider)
        sents_combined = sents_in + sents_on
        
        logits, cache = model.run_with_cache(sents_combined)
        detached_cache = {}
        for k in cache:
            detached_cache[k] = cache[k].detach().cpu().numpy()

        store_caches[prompt] =  detached_cache
        #logit_diff_results[prompt] = get_avg_logit_diff(logits, answ_tokens).item()
        print(f"Getting {method} logit difference for {prompt}")
        if method == "residual":
                # get accumulated residual
                accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
                logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, cache, logit_diff_directions)
                logit_lens_dict[prompt] = logit_lens_logit_diffs.detach().cpu().numpy()
                        
        elif method == "layers":
                per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
                per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache, logit_diff_directions)
                logit_lens_dict[prompt] = per_layer_logit_diffs.detach().cpu().numpy()
        else:
            print("Invalid type argument, please pass one between 'residual' and 'layers'")
            exit
        
        del cache; del logits

    #save dataframe and plots
    print(f"\nSaving Dataframe and plots")
    df = pd.DataFrame(logit_lens_dict, index = labels)
    plt.figure(figsize=(14,8))
    ax = sns.lineplot(df)
    if method == "residual":
        ax.set_title(f"{args.model} residual stream logit lens")
        ax.set_xlabel(f"Residual stream location")
    elif method == "layers":
        ax.set_title(f"{args.model} transformer layer logit lens")
        ax.set_xlabel(f"Layer")
    ax.set_ylabel("Logit difference")
    plt.xticks(rotation=45);

    ax.get_figure().savefig(f"{output_dir}/{args.model}_{method}_loglens.pdf")
    df.to_csv(f"{output_dir}/{args.model}_{method}_loglens.tsv", sep = "\t")
    print(f"\nResults and plots saved at {output_dir}")
    print(f"GPU memory available at the moment: {torch.cuda.mem_get_info()[0]/1000000000}/ {torch.cuda.mem_get_info()[1]/1000000000}")
 


    