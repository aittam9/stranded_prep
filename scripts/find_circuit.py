from functools import partial

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 
import pickle 
import os
import argparse

from src.utils import load_triplets, filter_triplets, prepare_sents, make_labels
from src.prompts import PROMPTS, models2try
# from src.eap_data_tools import EAPDataset, collate_EAP, get_logit_positions, logit_diff, load_model_eap, make_eap_input_df

from typing import Literal, List, Dict, Union, Tuple, Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

##############----------------------------adapted from EAP-IG library #####################
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

def get_nodes_edges_values(g):
    #edges and nodes involved to reach 85% of performance (make a funtcion)
    included_edges = g.count_included_edges()
    pct_edges_used = round((included_edges/ len(g.edges.keys()))*100, 2)
    nodes_included = g.count_included_nodes()
    pct_nodes_included = round((nodes_included/ len(g.nodes.keys()))*100, 2)
    return included_edges, pct_edges_used, nodes_included, pct_nodes_included

def make_stats_df(values):
    columns = ["prompt", "baseline", "circuit_perf", "pct_retained_perf","edges_included", "edges_included_pct", "nodes_included", "nodes_included_pct"]
    assert len(values) == len(columns), "Number of column values and labels must match!"
    df = pd.DataFrame(values).T
    df.columns = columns
    df = df.set_index("prompt")
    return df


def isolate_circuit(model, g, dataloader, metric, prompt, steps = range(0, 30000, 1000),
                     threshold = 85, method: Literal["best_circ", "trend"] = "best_circ"):
    """ 
    Perform the evaluation of the circuits retaining progressively less nodes depending on the range.
    Args:
    steps = the range and steps to follow to progressively throw out nodes 
    threshold = the percentage of model performance we want to be retained by the circuit
    method = whether returning the values of the threshold-performing circuit or the whole trend (useful to plot)

    Return:
    baseline, results, g if best_circ = the baseline value, the results of the circuit performing around the threshold and the scored graph
    baseline, performances if trend  = baseline value and dictionary of step (nodes included):results (metric performance)
    """
    performances = {}
    #compute baseline and instantiate a treshold
    print(f"Evaluating baseline for prompt {prompt}")
    baseline = evaluate_baseline(model, dataloader, partial(metric, loss=False, mean=False)).mean().item()
    # steps = range(0, 30000, 1000)
    # treshold = round(85 * baseline / 100, 2) + 0.01
    treshold = round(threshold * baseline / 100, 2) + 0.01
    print(f"Evaluating graph retaining edges in the  {steps}, method = {method}")
    for step in reversed(steps):
        topk = int(step)
        g.apply_topn(topk, True)
        results = evaluate_graph(model, g, dataloader, partial(metric, loss=False, mean=False)).mean().item()
        performances[step] = [results]
        if round(results, 2) <= treshold: #TODO elaborate on that and find a way to isoalte the smallest circuit possible 
            if method == "best_circ":
                print(f"The topk for 85% of the preformance is reached around {topk} retained edges")
                return baseline, results, g
    return baseline, performances

def save_circuit(g, dir_path, file_name:str, saving_method: Literal["pt","json"] = "pt"):
    file_name = f"{file_name}_circuit.pt"
    file_path = os.path.join(dir_path,file_name)
    if saving_method == "pt":
        g.to_pt(file_path)
    elif saving_method == "json":
        g.to_json(file_path)
    print(f"Graph saved at {file_path}")
    pass

#TODO to get more efficient memory usage
def del_garabage():
    pass

if __name__ == "__main__":
    #cli args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type= str, help = "The model to use")
    parser.add_argument("--method", "-met", type = str, default= "best_circ", help = "Wehter to find a threshold performing circuit or save the whole trend of performance edge wise")
    parser.add_argument("--threshold", "-t", type = int, default = 86, help = "The threshold to use to isolate the circuit. Will be used only when method == best_circ")
    parser.add_argument("--decrease_factor", "-dfac", type = int, nargs = "+", default = [1000, 30000, 1000], help = "the value to follow as upper bound to include edges and moving window to throw out edges" )
    args = parser.parse_args()
    
    #unpack the input decrease factor to be used as a reference to set upper bound of incuded nodes as well as 
    #the sliding window to decrease it
    start, stop, step = args.decrease_factor
    decreasing_fatcor = range(start, stop, step)
    
    #handle model mispelling
    if not args.model in models2try:
        print(f"Model not available, choose another one")
        exit()

    #create dedicated directories to store results
    base_dir = "../stranded_prep"
    global_outdir = f"circuits/{args.model}"
    circuits_subdir = f"{global_outdir}/pt_circuits"
    if not os.path.exists(global_outdir):
        os.mkdir(os.path.join(base_dir,global_outdir))
        print(f"Created global_outdir at {global_outdir}")
    if not os.path.exists(circuits_subdir):
        os.mkdir(os.path.join(base_dir,circuits_subdir))
        print(f"Created circuit subdir at {circuits_subdir}")
    
    #load model and raw data
    model_name = models2try[args.model]
    model = load_model_eap(model_name)
    triplets = load_triplets("./data/triplets_in_on.csv")
    triplets2consider = filter_triplets(triplets, model.tokenizer)
    
    # iterate over the prompts to get results 
    print(f"Starting analysis, method = {args.method}")
    all_prompt_dfs = []
    for prompt in PROMPTS:
        eap_input_df = make_eap_input_df(PROMPTS[prompt], triplets2consider, model)
        #convert the df in dataset and dataloader
        ds = EAPDataset(eap_input_df)
        dataloader = ds.to_dataloader(10)

        # Instantiate a graph with a model
        g = Graph.from_model(model)
        # Attribute using the model
        print(f"Attributing model for {prompt}")
        attribute(model, g, dataloader, partial(logit_diff, loss=False, mean=True), method='EAP-IG-inputs', ig_steps=5)
        #store the graph
        
        # #compute baseline and instantiate a treshold
        # print(f"Evaluating baseline for prompt {prompt}")
        # baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
        # #TODO maybe factorize in a function
        # steps = range(0, 30000, 1000)
        # treshold = round(85 * baseline / 100, 2) + 0.01
        # print(f"Evaluating graph retaining edges in the  {steps}")
        # for step in reversed(steps):
        #     topk = int(step)
        #     g.apply_topn(topk, True)
        #     results = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
        #     if round(results, 2) <= treshold:
        #         print(f"The topk for 85% of the preformance is reached around {topk} retained edges")
        #         break
        #method
        if args.method == "best_circ":
            
            baseline, results, g = isolate_circuit(model, g, dataloader, logit_diff, prompt, steps = decreasing_fatcor, threshold = args.threshold, method = "best_circ")
            included_edges, pct_edges_used, nodes_included, pct_nodes_included  = get_nodes_edges_values(g)
            actual_pct = round(results/ baseline *100, 2)
            values = [prompt, round(baseline, 2), round(results,2), actual_pct, included_edges, pct_edges_used, nodes_included, pct_nodes_included]
            #make the df for the template and append it to global df
            stats_df = make_stats_df(values)
            all_prompt_dfs.append(stats_df)

            #save circuit
            save_circuit(g, circuits_subdir, prompt)
        
        elif args.method == "trend":
            baseline, results = isolate_circuit(model, g, dataloader, logit_diff, prompt, steps = decreasing_fatcor, threshold = args.threshold, method = "trend")
            results["baseline"] = [baseline]
            results["prompt"] = [prompt]
            results_df = pd.DataFrame(results).set_index("prompt")
            all_prompt_dfs.append(results_df)

            #check if the df is being built correctly
            if prompt == list(PROMPTS.keys())[1]:
                print(pd.concat(all_prompt_dfs, axis = 0))
        

        # included_edges, pct_edges_used, nodes_included, pct_nodes_included  = get_nodes_edges_values(g)

        # #prepare values for df (maybe trun it into a dict)
        # values = [prompt,
        #         round(baseline, 2), 
        #         round(results,2),included_edges,
        #         pct_edges_used, 
        #         nodes_included,
        #         pct_nodes_included]
        
        # stats_df = make_stats_df(values)
        # all_prompt_dfs.append(stats_df)
    
    #save the complete output df
    complete_df = pd.concat(all_prompt_dfs, axis = 0)
    outfile_path = f"{global_outdir}/circuit_{args.method}.tsv"
    complete_df.to_csv(outfile_path, sep = "\t")
    print(f"Saved dataframe with {args.method} stats saved at {outfile_path} ")


  

    
    