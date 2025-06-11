import os 
import numpy as np
import pandas as pd
from eap.graph import Graph 
import matplotlib as plt 
import seaborn as sns 
import argparse
import seaborn as sns



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type= str, help = "The model to use")
    args = parser.parse_args()

    indir = f"../circuits/{args.model}/pt_circuits"
    outdir = f"../circuits/{args.model}"
    plots_outdir = f"../circuits/{args.model}/plots"
    if not os.path.exists(plots_outdir):
        os.mkdir(plots_outdir)
        print(f"Created directory {plots_outdir}")
    
    #load circuits into graphs
    circuits = {}
    for el in os.listdir(indir):
        name = el.split(".")[0]
        circuits[name] = Graph.from_pt(os.path.join(indir,el))
        print(f"Loaded {name}")

    #get only the edges included into each graph
    only_ingraph_edges= {}
    for k in circuits:
        only_ingraph_edges[k] = [n for n in circuits[k].edges if circuits[k].edges[n].in_graph ]

    #empty matrix to store IoU and edge recall values
    iou_matrix = np.zeros((5,5))
    er_matrix = np.zeros((5,5))
    #compute iou values over edges of each template pair (maybe do a function)
    for n,k in enumerate(only_ingraph_edges):
        for j,k2 in enumerate(only_ingraph_edges):
            a,b = set(only_ingraph_edges[k]), set(only_ingraph_edges[k2])
            
            intersection = a.intersection(b)
            union = a.union(b)
            
            iou = len(intersection) / len(union)
            iou_matrix[n,j] = iou
            
            edge_recall = len(intersection) / len(b)
            er_matrix[n,j] = edge_recall

            print(f"The IoU between {k} and {k2} is {round(iou,2)}")
        print("\n----------------------------------------------------------------------------------------")

    #dataframe IoU
    labels = list(only_ingraph_edges.keys())
    iou_df = pd.DataFrame(iou_matrix, columns = labels, index = labels)
    iou_df.to_csv(f"{outdir}/inter_prompt_iou.csv")
    print(f"Dataframe saved!")

    #plot IoU
    avg_iou = np.mean(iou_matrix).round(2)
    fig = sns.heatmap(iou_df, cmap = "Blues", cbar = False, annot= True)
    plt.title(f"IoU of different templates circuits (Avg {avg_iou})")
    plt.tight_layout()
    plt.show()
    fig.get_figure().savefig(f"{plots_outdir}/inter_prompt_iou.png")
    print(f"Figure saved!")

    #dataframe edge recall
    labels = list(only_ingraph_edges.keys())
    er_df = pd.DataFrame(er_matrix, columns = labels, index = labels)
    er_df.to_csv(f"{outdir}/inter_prompt_er.csv")
    print(f"Dataframe saved!")

    #plot edge recall
    avg_er = np.mean(er_matrix).round(2)
    fig = sns.heatmap(er_df, cmap = "Blues", cbar = False, annot= True)
    plt.title(f"EdgeRecall of different templates circuits (Avg {avg_er})")
    plt.tight_layout()
    plt.show()
    fig.get_figure().savefig(f"{plots_outdir}/inter_prompt_er.png")
    print(f"Figure saved!")



    