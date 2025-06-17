from transformer_lens import HookedTransformer
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.utils import load_triplets, filter_triplets, make_labels, prepare_sents
from src.prompts import models2try, PROMPTS
from typing import List, Literal
import argparse 
import os
import pickle 

#helper to plot the pca #TODO adjuest the legend and suptitles
def plot_pca(df_list, hue:str = "Landmark type"):
    fig, axes = plt.subplots(4,4, figsize = (14, 12))
    n = -1
    xlabel = ""
    ylabel = ""
    legend = False
    for row in range(4):
        for col in range(4):
            n += 1
            if n == 15:
                legend = True
            if row == 3:
                xlabel = "PC1"
            # if  n in [0,4,8,12]:
            #     ylabel = "PC2"
            sns.scatterplot(data = df_list[n], ax=axes[row,col], x = "PC1", y = "PC2", hue = hue, alpha = 0.5, legend = legend).set(title = f"Layer {n}", xlabel = xlabel, ylabel = ylabel)
            plt.subplots_adjust(wspace=.3, hspace=.5)
    plt.suptitle("Last token PCA projection at each layer")
    plt.tight_layout();
    return fig

#helper to fit the pca
def fit_pca(activations, concat = False, include_prompt_labels = False):
    """ Fit a PCA on each layer and return a list of dataframes with principal components.
    Args:
        activations: the activations to transform/reduce
        cocat: wheter the activations are the concatenation of all prompts or a single prompt
        include_prompt_labels: whether to include a label signalling the prompt fom which te PCs are derived
    Return:
        a list of dataframes with PCs
    """
    if concat:
        length = len(PROMPTS)
    else:
        length = 1
    all_transformed_df = []
    #fit pca on each layer and plot it
    for layer in range(activations.shape[0]):
        X = activations[layer].detach().cpu().numpy()
        pca = PCA(n_components=2)
        pca.fit(X)
        X_transform = pca.transform(X) 
        df_transform= pd.DataFrame(X_transform, columns = ["PC1", "PC2"])
        df_transform["Landmark type"] = (["Container"]* 152 + ["Surface"] * 152)* length #mulitply by 5 if it is the concatenation of all prompts
        if include_prompt_labels:
            template_labels = []
            for k in PROMPTS:
                template_labels.extend([k]*len(triplets2consider)*2)
            df_transform["prompt"] = template_labels
        all_transformed_df.append(df_transform)
    return all_transformed_df


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type= str, help = "The model to use")
    parser.add_argument("--template", "-t", choices = list(PROMPTS.keys()) + ["all"] + ["each"], default = "all", help = "Decide if to apply pca on a given templates or on all of them. Default = all, will concatenate all the data points")
    #parser.add_argument("--plot_type", "-pt", options = ["Landmark type", "prompt"], default = "Landmark type")
    parser.add_argument("--make_plot", "-mp", action = "store_true", help = "Load an already transformed dataset and make the plot for the target templates. Skip the the projection.")
    args = parser.parse_args()
    
    #TODO instantiate output directories
    outdir = f"./pca/{args.model}/"
    plots_outdir = f"./pca/{args.model}/plots"
    if not os.path.exists(plots_outdir):
        os.makedirs(plots_outdir)
        print(f"Created directory {plots_outdir}")
    
    if not args.make_plot:
        # if the script is not launched for visualization only, extract activations, fit pca and save components
        # load model and data
        model_name = models2try[args.model]
        model = HookedTransformer.from_pretrained(model_name, cache_dir = "/extra/mattia.proietti/tl_models")
        triplets = load_triplets("./data/triplets_in_on.csv")
        triplets2consider = filter_triplets(triplets, model.tokenizer)
        # answ_tokens = make_labels(model.tokenizer, triplets2consider).to(model.cfg.device)
    
        tensor_sequence = []
        print("Extracting activations...")
        for prompt in tqdm(PROMPTS):
            torch.cuda.empty_cache()
            each_block_last_tok_act = torch.zeros(size = (model.cfg.n_layers, len(triplets2consider*2), model.cfg.d_model)).to(model.cfg.device)
            sents_in, sents_on = prepare_sents(PROMPTS[prompt], triplets2consider)
            #get activation cache
            logits, cache = model.run_with_cache(sents_in+ sents_on)
            #extract last token final residual stream at each layer
            for layer in range(model.cfg.n_layers):
                each_block_last_tok_act[layer] += cache[f"blocks.{layer}.hook_resid_post"][:,-1,:]
            tensor_sequence.append(each_block_last_tok_act)
            #delete cache to save memory 
            for k in cache:
                p = cache[k].detach().cpu().numpy()
                del p
        #concatenate the tensor along input_sentence dimension
        concat_tensor = torch.cat(tensor_sequence, dim = 1)
        assert concat_tensor.shape[1] == len(triplets2consider*2)*len(PROMPTS)

        #fit the pca on all layers
        print("Fitting PCA...")
        all_transformed_df = fit_pca(concat_tensor, concat = True, include_prompt_labels=True)
        #all_transformed_df.to_csv(os.path.join(outdir, "all_template_transforms.tsv", sep = "\t"))
        with open(os.path.join(outdir, "all_template_transforms.pkl"), "wb") as outfile:
            pickle.dump(all_transformed_df, outfile)
        print(f"Projections saved at {outdir}")
        
        #plot the pca with landmark as hue
        fig = plot_pca(all_transformed_df)
        fig.savefig(os.path.join(plots_outdir, "pca_all_landmark_hue.png"))  

        # plot pca with prompt as hue
        fig2 = plot_pca(all_transformed_df, hue="prompt")
        fig2.savefig(os.path.join(plots_outdir, "pca_all_prompt_hue.png"))
        print(f"Figures saved at {plots_outdir}")
    
    # to be improved
    elif args.make_plot:
        try:
            # all_transformed_df = pd.read_csv(os.path.join(outdir, "all_template_transforms.tsv"), sep = "\t")
            all_transformed_df = pickle.load(open(os.path.join(outdir, "all_template_transforms.pkl"), "rb"))
        except:
            print(f"The projections file does not exists. Create on fitting a PCA first.")
            exit()
        
        if args.template == "each":
            for k in PROMPTS:
                target_dfs = [df[df["prompt"] == k] for df in all_transformed_df]
                fig = plot_pca(target_dfs)
                file_name = f"pca_{k}.png"
                fig.savefig(os.path.join(plots_outdir, file_name ))
            print(f"Figures saved at {plots_outdir}")
        
        elif args.template == "all":
            #plot the pca with landmark as hue
            fig = plot_pca(all_transformed_df)
            fig.savefig(os.path.join(plots_outdir, "pca_all_landmark_hue.png"))  

            # plot pca with prompt as hue
            fig2 = plot_pca(all_transformed_df, hue="prompt")
            fig2.savefig(os.path.join(plots_outdir, "pca_all_prompt_hue.png"))

        elif args.template in list(PROMPTS.keys()):
            target_dfs = [df[df["prompt"] == args.template] for df in all_transformed_df]
            fig = plot_pca(target_dfs)
            fig.savefig(os.path.join(plots_outdir, f"pca_{args.template}.png"))
    else:
        print("invalid arguments")


   