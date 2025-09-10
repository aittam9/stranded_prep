import csv
import gc
import json
import random
import argparse 
import os

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from transformers import  AutoTokenizer, AutoModelForCausalLM
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
# from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score 

from src.utils import filter_triplets, prepare_sents, load_triplets 
from src.prompts import PROMPTS, models2try 



#helper function to extract hidden states for a given word id
def extract_hidden_states(model_outptut,inputs, target_word_id, tokenizer, labels):
  target_hidden_states = {}
  target_words_string = [tokenizer.decode(i[target_word_id]).strip() for i in  inputs["input_ids"]]
  targets = torch.stack(model_outptut["hidden_states"])[:,:,target_word_id,:]
  print(targets.shape)

  assert targets.shape[1] == len(labels) == len(target_words_string), "Inconsistent lengths between inputs and labels"
  target_hidden_states["inputs"] = targets.detach().cpu().numpy()
  target_hidden_states["words"] = target_words_string
  target_hidden_states["labels"] = labels

  return target_hidden_states

# get probing results from given activations TODO: modify to do crossvalidation
def get_probe_res(hidden_states, prompt, n_layers, cv = False):
  scaler = StandardScaler()
  n_layers = n_layers + 1 #plus one to add the embedding layer
  metrics = {}
  split_size = 0.20
  for i in range(n_layers):
    X = hidden_states[prompt]["inputs"][i]
    y = np.array(hidden_states[prompt]["labels"])
    words = hidden_states[prompt]["words"]
    if not cv:
      #standard probing (no cv)
      X, y, words = shuffle(X, y, words, random_state=42)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
      X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

      clf = LogisticRegression(random_state=0, max_iter= 150).fit(X_train, y_train)
      score = clf.score(X_test, y_test)

    elif cv:
      #cross validation
      clf = LogisticRegression(random_state=0, max_iter= 200)
      scores = cross_val_score(clf, X, y, cv=5)
      score = scores.mean()

    if i== 0:
      layer = f"Embedding"
    else:
      layer = f"Layer_{i}"

    metrics[layer] = {}
    metrics[layer]["Accuracy"] = score
    # metrics[layer]["Precision"] = precision_score(y_test, clf.predict(X_test))
    # metrics[layer]["Recall"] = recall_score(y_test, clf.predict(X_test))
    # metrics[layer]["F1"] = f1_score(y_test, clf.predict(X_test))
  return metrics


#plot probe results TODO: modify to account for line plot
def plot_probe_res(metric_results, random_states = False, plot_type = "bar"):

  figure, axes = plt.subplots(3,2, figsize = (20,10))
  axes = axes.flatten()
  #color
  if random_states:
    color = "red"
  else:
    color = "blue"
  for n,p in enumerate(PROMPTS):
    axes[n].set_ylim(0,1)
    axes[n].set_title(f"Probing Accuracy for {p}")
    metric_df = pd.DataFrame(metric_results[p]).round(2)
    metric_df.T["Accuracy"].plot(kind = "bar", ax = axes[n], color = color )
    if not n in [3,4]:
      axes[n].set_xticks([])
  figure.delaxes(axes[5])
  return figure, axes

# utils to make a long form df to plot results
def make_long_df(results_dict, target_types = ["Landmark", "Random after landmark", "Last token"]):
  # prepare dataframe for plotting
  all_rows = []
  for p in results_dict:
    for t in target_types:
      for l in results_dict[p][t]:
        row = (p, t, l, results_dict[p][t][l]["Accuracy"])
        all_rows.append(row)

  long_df = pd.DataFrame(all_rows, columns = ["Template", "Target Word", "Layer", "Accuracy"])
  return long_df


DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # TODO add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "llama3.2-1b", help = "model to use for probing")
    args = parser.parse_args()

    
    # TODO make directories for current model
    output_dir = f"../probing/{args.model}/"
    plots_dir = f"../probing/{args.model}/plots/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    #load model
    model_id = models2try[args.model] #"meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map = DEVICE, cache_dir = "/extra/mattia.proietti/hf_models")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    #load triplets and prepare labels
    triplets = load_triplets("./data/triplets_in_on.csv")
    triplets2consider = filter_triplets(triplets, tokenizer)
    labels = [0] * len(triplets2consider) + [1] * len(triplets2consider)

    #extract activations
    target_hidden_states = {}
    random_word_hidden_states = {}
    random_word_before_target_hs  = {}
    random_word_after_target_hs = {}
    last_word_hidden_states = {}
    for prompt in PROMPTS:
        target_hidden_states[prompt] = {}
        random_word_hidden_states[prompt] = {}

        tokenized_template = tokenizer.tokenize(PROMPTS[prompt][0])
        target_word_id = tokenized_template.index("Ġ{}") + 1

        sents_in, sents_on = prepare_sents(PROMPTS[prompt], triplets2consider)
        complete_sents = sents_in + sents_on
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(complete_sents, return_tensors="pt", padding=True).to(model.device)

        print(f"running model on {prompt}")
        with torch.inference_mode():
            out = model(**inputs,  output_hidden_states = True)

        #get landmark hidden states
        # target_word_id = tokenizer.tokenize(prompts[prompt][0]).index("Ġ{}") + 1
        target_hidden_states[prompt] = extract_hidden_states(out, inputs, target_word_id, tokenizer, labels)

        #get random word before target hidden states
        random_id_before_target = random.choice(list(range(len(tokenized_template)))[1: target_word_id])
        random_word_before_target_hs[prompt] = extract_hidden_states(out, inputs, random_id_before_target, tokenizer, labels)

        #get random word after target hidden states
        random_id_after_target = random.choice(list(range(len(tokenized_template)))[target_word_id+1:-1])
        random_word_after_target_hs[prompt] = extract_hidden_states(out, inputs, random_id_after_target, tokenizer, labels)

        #get last token hidden states
        last_word_hidden_states[prompt] = extract_hidden_states(out, inputs, -1)

    print("--Extracted all desired hidden states--")
    # probing 
    target_types = ["Landmark", "Random after landmark", "Last token"]
    all_prompt_res = {}
    for prompt in PROMPTS:
        all_prompt_res[prompt] = {}
        print("\nProbing for: {prompt}")
        for target in target_types:
            print(f"Probing inputs from {prompt} for target word: {target}")
            if target == "Landmark":
                all_prompt_res[prompt][target] = get_probe_res(target_hidden_states, prompt, model.config.num_hidden_layers, cv = True)
            elif target == "Random after landmark":
                all_prompt_res[prompt][target] = get_probe_res(random_word_after_target_hs, prompt, model.config.num_hidden_layers, cv = True)
            elif target == "Last token":
                all_prompt_res[prompt][target] = get_probe_res(last_word_hidden_states, prompt, model.config.num_hidden_layers, cv = True)

    # plot results 
    # make the long-form df for plotting
    long_df  = make_long_df(all_prompt_res)
    long_df.to_csv(f"{output_dir}/probing_results_all_targets.csv", index = False)

    # plot results
    sns.set_style("darkgrid")
    plt.figure(figsize = (10,5))
    fig = sns.lineplot(data = long_df, x = "Layer", y = "Accuracy", hue = "Target Word", style = "Target Word", markers = True, dashes = False).set(ylim = (0,1), title = "Probing Accuracy comparison using different words in the templates", ylabel = "Accuracy", xlabel = "Layer")
    plt.xticks(rotation=45);
    #save figure
    fig.get_figure().savefig(f"{plots_dir}/probing_all_targets.png")