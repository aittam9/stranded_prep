from functools import partial
import os
import pandas as pd
import numpy as np

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 

import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm

from src.utils import load_triplets, filter_triplets, prepare_sents, make_labels
from src.prompts import PROMPTS,models2try
from src.eap_data_tools import EAPDataset, collate_EAP, get_logit_positions, logit_diff, load_model_eap, make_eap_input_df

import argparse 
import pickle


if __name__ == "__main__":
    """
    Compute the cross-template performance of each circuits (cross-template faithfulness).
    Compute also the performance of a common circuit obtained selecting the edges at the intersection of all templates circuits, against each template.
    If --only_core is flagged only the latter will take place
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type= str, help = "The model to use")
    parser.add_argument("--only_core", "-oc", action="store_true", help = "allows to only compute the intersection across circuits")
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
        if el.endswith("pt"):
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

    # circuit-circuit faithfulness
    if not args.only_core:
        print(f"Cross template-circuit faithfulness among each circuit")
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
        
        #plots and save plots #TODO try to move ticks parameters to a golabal setting
        fig = sns.heatmap(res_df.iloc[:,:-1], cbar = False, cmap = "Blues", annot = True)
        plt.title(f"Cross-template circuits performance")
        fig.tick_params(left=False, bottom=False)
        plt.tight_layout()
        #save figure
        fig.get_figure().savefig(f"{plots_outdir}/cross_template_faith.png")
        print(f"Figure 1 saved at {plots_outdir}")

        # Transform performances to ratio against template-derived circuit
        res_df = res_df.iloc[:,:-1]
        new = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                if i== j:
                    gold = res_df.iloc[i,j] 
                pct = res_df.iloc[i,j]/gold
                new[i,j] = round(pct,2) 
        # make df with ratios
        df_pct = pd.DataFrame(new, columns = res_df.columns[:-1], index = res_df.index) 
        fig2 = sns.heatmap(df_pct, cmap = "Blues", annot = True, cbar = False)
        fig2.set(title = "Cross-template circuit performance (against target)")
        fig2.tick_params(left=False, bottom=False)
        plt.tight_layout()
        # save figure
        fig2.get_figure().savefig(f"{plots_outdir}/cross_template_faith_pct.png")
        print(f"Figure 2 (pct) saved at {plots_outdir}")


    # evaluate only-core circuit against each template
    print(f"\nComputing faithfulness of the core circuit against all templates")
    # Load the saved core edges into an empty graph instantiated from model
    print(f"Loading core edges into empty graph...")
    core_edges = pickle.load(open(os.path.join(circuits_indir, "core_circuit_edges.pkl"),"rb"))
    core_g = Graph().from_model(model)
    for edge in tqdm(core_edges):
         core_g.edges[edge].in_graph = True
    print(f"Total edges in graph {core_g.count_included_edges()}")
    
    #instantiate an empty vector to store results and loop over the prompt to evaluate the core circuit
    res_vector = np.zeros(5)
    index = [] #to maintain an index for the df
    for j,t in enumerate(PROMPTS):
                print(f"\nAssessing core_circuits on {t}")
                # format inputs for eap df
                eap_input_df = make_eap_input_df(PROMPTS[t], triplets2consider, model)
                #convert the df in dataset and dataloader
                ds = EAPDataset(eap_input_df)
                dataloader = ds.to_dataloader(10)
                results = evaluate_graph(model, core_g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
                res_vector[j] = round(results,2)
                #track previous circuit performance and baseline performance
                model_baseline = baselines_df[baselines_df["prompt"] == t]["baseline"].item()
                previous_circuit_baseline = baselines_df[baselines_df["prompt"] == t]["circuit_perf"].item()
                print(f"The circuit's performance is {results}\nFull model baseline was:{model_baseline}\nTarget Template performance was: {previous_circuit_baseline}")
                index.append(t)
    
    #build dataframe and save it
    res_vector_df = pd.DataFrame(res_vector, index = index, columns = ["core_circuit"]).T
    res_vector_df.to_csv(f"{outdir}/core_circ_cross_template_faith.tsv", sep = "\t")
    print(f"Core circuit result dataframe saved at {outdir}")
    
    #plots
    fig3 = sns.heatmap(res_vector_df, square = True, cbar = False, cmap = "Blues", annot=True)
    fig3.tick_params(left=False, bottom=False)
    fig3.set(title = "Performance of the Core Circuit against each template")
    fig3.get_figure().savefig(f"{plots_outdir}/core_circ_cross_template_faith_pct.png")
    print(f"Core circuit performance figure saved at {plots_outdir}")
