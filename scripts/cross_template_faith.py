from functools import partial
import os
import pandas as pd
import numpy as np

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 

import matplotlib.pyplot as plt 
import seaborn as sns 

from src.utils import load_triplets, filter_triplets, prepare_sents, make_labels
from src.prompts import PROMPTS,models2try
from src.eap_data_tools import EAPDataset, collate_EAP, get_logit_positions, logit_diff, load_model_eap, make_eap_input_df

import argparse 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type= str, help = "The model to use")
    args = parser.parse_args()
    
    
    outdir = f"./circuits/{args.model}/"
    circuits_indir = f"./circuits/{args.model}/pt_circuits"
    
    plots_outdir = f"./circuits/{args.model}/plots"
    if not os.path.exists(plots_outdir):
        os.mkdir(plots_outdir)
        print(f"Created directory {plots_outdir}")

    baselines_path = f"./circuits/{args.model}/circuit_best_circ.tsv"
    baselines_df = pd.read_csv(baselines_path, sep = "\t")
    
    #load model and raw data
    model_name = models2try[args.model]
    model = load_model_eap(model_name)
    triplets = load_triplets("./data/triplets_in_on.csv")
    triplets2consider = filter_triplets(triplets, model.tokenizer)

    #load circuits 
    unord_circuits = {}
    for el in sorted(os.listdir(circuits_indir)):
        name = el.split(".")[0]
        unord_circuits[name] = Graph.from_pt(os.path.join(circuits_indir,el))
        print(f"Loaded {name}")
    
    #align circuits names with templates, so that we are sure to have the same ordering
    order = ['templates0_circuit',
    'templates_questions_circuit',
    'templates_questions2_circuit',
    'templates_relative_neg_circuit',
    'templates_relatives_aff_circuit']
    circuits = {}
    for n,k in enumerate(unord_circuits):
        print(order[n])
        circuits[order[n]] = unord_circuits[k]
    assert list(circuits.keys()) == order, "Prompts and keys order don't match"

    res_matrix = np.zeros((5,5))
    model_c1_c2_comp = {}
    for i,c in enumerate(circuits):
        g = circuits[c]
        for j,t in enumerate(PROMPTS):
            print(f"\nAssessing {c} on {t}")
            # format inputs for eap df
            eap_input_df = make_eap_input_df(PROMPTS[t], triplets2consider, model)
            #convert the df in dataset and dataloader
            ds = EAPDataset(eap_input_df)
            dataloader = ds.to_dataloader(10)
            
            results = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
            res_matrix[i,j] = round(results,2)
            #track previous circuit performance and baseline performance
            model_baseline = baselines_df[baselines_df["prompt"] == t]["baseline"].item()
            previous_circuit_baseline = baselines_df[baselines_df["prompt"] == t]["circuit_perf"].item()
            print(f"The circuit's performance is {results}\nFull model baseline was:{model_baseline}\nTarget Template performance was: {previous_circuit_baseline}")
        
    #save results
    columns = baselines_df["prompt"].to_list()
    index = list(circuits.keys())
    average = np.mean(res_matrix).round(2)
    res_df = pd.DataFrame(res_matrix, columns = columns, index = index)
    res_df["baseline"] = baselines_df["baseline"]
    res_df.to_csv(f"{outdir}/cross_template_faith.tsv", sep = "\t")
    print(f"Dataframe saved at {outdir}")
    
    #plots and save plots
    
    fig = sns.heatmap(res_df.iloc[:,:-1], cbar = False, cmap = "Blues", annot = True)
    plt.title(f"Cross-template circuits performance")
    plt.tight_layout()

    #save figure
    fig.get_figure().savefig(f"{plots_outdir}/cross_template_faith.png")
    print(f"Dataframe saved at {plots_outdir}")