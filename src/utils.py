from functools import partial
import csv
import einops
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch
#from circuitsvis.attention import attention_heads
from fancy_einsum import einsum
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import  HookedTransformer
import matplotlib.pyplot as plt
#from neel_plotly import line, imshow, scatter
import gc 


def load_triplets(path):
    triplets = [i for i in csv.reader(open(path, "r") )if i][1:]
    triplets = list(map(lambda x: [i.lower() for i in x], triplets))
    return triplets

def filter_triplets(triplets, tokenizer):
    """
      Filter out all the triplets that are tokenized to different lengths, to keep an equal number of tokens in each batch.
      Args:
      triplets: the list of loaded triplets
      tokenizer: the model tokenizer to use
    """
    filtered_triplets = []
    triplets_removed = 0
    for t in triplets:
      if t:
        new_t = [" "+i for i in t]

        tknzed_t = [tokenizer(i, add_special_tokens = False) for i in new_t]
        if len(tknzed_t[0].input_ids) == len(tknzed_t[1].input_ids) == len(tknzed_t[2].input_ids):
          filtered_triplets.append(t)
        else:
          triplets_removed+=1
          pass

    print(f"Number of triplets removed: {triplets_removed}")
    return filtered_triplets

def get_accuracy(logits, labels) ->int : 
    """"Compute all the time the model actually gives the target label as output
     Args;
      logits: the logits computed in a previous step
      labels: the labels encoded as tensors
    Return:
      accuracy
       """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels[:,0]).sum().item()
    total = len(labels)
    accuracy = (correct / total) *100
    return accuracy

def get_relative_accuracy(logits, labels):
      """
        Compute the number of times the model prefer (assign higher logits to) the target token  over the counterfactual one, normalized over the len of inputs.
      """
      if len(logits.shape) ==3:
        logits = logits[:,-1]
      correct_logits = logits.gather(1, labels[:, 0].unsqueeze(1))
      incorrect_logits = logits.gather(1, labels[:, 1].unsqueeze(1))
      return ((correct_logits > incorrect_logits).sum().item() / labels.shape[0]) * 100

def prepare_sents(template_pair, triplets2consider):
    """
      Prepare the sentences to be fed to the model formatting them in the template.
      Arguments:
      template_pair: the template to format the triplets
      triplet2consider: the triplets filtered on the basis of a model tokenizer

      Return:
      sents_in, sents_on: two lists containing respectively formatted sentences expecting in and sents expecting on
    """
    template_in, template_on = template_pair
    sents_in = [" ".join(template_in.format(e[1],e[0]).split()[:-1]) for e in triplets2consider]
    sents_on = [" ".join(template_on.format(e[2],e[0]).split()[:-1]) for e in triplets2consider]
    return sents_in, sents_on


def get_avg_logit_diff(logits, answ):
  """
      Get the average logit difference between the correct and incorrect answers.
  """
  if len(logits.shape) ==3:
     logits = logits[:,-1]
  correct_logits = logits.gather(1, answ[:, 0].unsqueeze(1))
  incorrect_logits = logits.gather(1, answ[:, 1].unsqueeze(1))
  return(correct_logits - incorrect_logits).mean()

#make labels from indexes
def make_labels(tokenizer, triplets2consider):
  """
      Prepare the targets to be used as prediction labels.
      Args:
      tokenizer: the model tokenizer to encode labels
      triplet2consider: the triplets already filtered with the same tokenizer

      Return:
      labels: a stacked tensor made by the labels
  """
  length = len(triplets2consider)
  in_id, on_id = tokenizer.encode(" in", add_special_tokens = False), tokenizer.encode(" on", add_special_tokens = False)
  labels = torch.tensor(in_id* length+ on_id*length)
  counter_logits = torch.tensor(on_id*length + in_id*length)
  return torch.stack((labels, counter_logits), dim = 1)

#helper to format results for plotting
def format_res4plot(results, mode = "logit_diff"):
  """ format accuracy results for plotting"""
  columns = ["model", "prompt", mode]
  rows = []
  for model in results:
    for prompt in results[model]:
      rows.append((model.split("/")[-1], prompt,round(results[model][prompt],2)))
  return pd.DataFrame(rows, columns = columns)


#get the logit lens logit diffs
#taken from neel nanda exploratory analysis tutorial
def residual_stack_to_logit_diff(residual_stack, cache, logit_diff_directions) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    #dot product and summation along all axis between the residual stack and the logit diff directions
    return einsum("... batch d_model, batch d_model -> ...", scaled_residual_stack, logit_diff_directions,) / logit_diff_directions.shape[0]