from transformers import  AutoTokenizer, AutoModelForCausalLM  
import pandas as pd
import torch
import csv
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import json

from transformers.utils.logging import disable_progress_bar
disable_progress_bar() 

from utils import load_triplets, filter_triplets, get_accuracy, get_avg_logit_diff, get_relative_accuracy, make_labels, prepare_sents, format_res4plot
from prompts import PROMPTS, models2try

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# def get_all_results(model_id,PROMPTS, triplets):
#     model = AutoModelForCausalLM.from_pretrained(model_id, device_map = DEVICE)
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     print(f"Using {model.device}")
#     #filter triplets
#     triplets2consider = filter_triplets(triplets, tokenizer)
#     logit_diff_results = {}
#     accuracy_results = {}
#     relative_accuracy_results = {}
#     #loop over prompts
#     for prompt in PROMPTS:
#             print(f"Getting metrics for {prompt}")
#             # format the sentences with the templates
#             sents_in, sents_on = prepare_sents(PROMPTS[prompt], triplets2consider)

#             #get input ids and and answer labels
#             tokenizer.pad_token = tokenizer.eos_token
#             inputs = tokenizer(sents_in + sents_on, return_tensors = "pt", padding = True).to(DEVICE)
#             answ_id = make_labels(tokenizer, triplets2consider).to(DEVICE)
            
#             #get model output
#             with torch.inference_mode():
#                 out = model(**inputs, max_new_tokens = 1)

#             #get last token logit and average logit difference
#             # logits = out.logits[:, -1]
#             logit_diff_results[prompt]  = get_avg_logit_diff(logits, answ = answ_id).item()
#             accuracy_results[prompt] = get_accuracy(logits, answ_id)
#             relative_accuracy_results[prompt] = get_relative_accuracy(logits, answ_id)

#             #save space
#             logits = logits.detach().cpu()
#             answ_id = answ_id.detach().cpu()
#             del logits
#             del answ_id
#             del model

#     return logit_diff_results, accuracy_results, relative_accuracy_results


if __name__ == "__main__":

    #TODO add the possibility to elicit metrics for all models or for one single model
    input_path = "./triplets_in_on.csv" # to add possibility to choose extended triplets
    triplets = load_triplets(input_path)
    
    
    all_models_all_results = {}
    logit_diff_results = {}
    accuracy_results = {}
    relative_accuracy_results = {}
    for m in models2try:
        #clean memory garbage
        torch.cuda.empty_cache()
        gc.collect()
        model_id = models2try[m]
        
        print(f"\nLoading {model_id}")
        #load the model
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map = DEVICE, cache_dir = "/extra/mattia.proietti/hf_models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Using {model.device}")

        logit_diff_results[model_id] = {}
        accuracy_results[model_id] = {}
        relative_accuracy_results[model_id] = {}

        #filter triplets
        triplets2consider = filter_triplets(triplets, tokenizer)
        #loop over prompts
        
        for prompt in PROMPTS:
            print(f"Getting metrics for {prompt}")
            # format the sentences with the templates
            sents_in, sents_on = prepare_sents(PROMPTS[prompt], triplets2consider)

            #get input ids and and answer labels
            tokenizer.pad_token = tokenizer.eos_token
            inputs = tokenizer(sents_in + sents_on, return_tensors = "pt", padding = True).to(DEVICE)
            answ_id = make_labels(tokenizer, triplets2consider).to(DEVICE)
            
            #get model output
            with torch.inference_mode():
                out = model(**inputs, max_new_tokens = 1)

            #get last token logit and average logit difference
            # logits = out.logits[:, -1]
            avg_logit_difference = get_avg_logit_diff(logits, answ = answ_id)
            print(f"Average logit difference for {prompt}: {avg_logit_difference}")

            # store results
            logit_diff_results[model_id][prompt] = avg_logit_difference.item()
            accuracy_results[model_id][prompt] = get_accuracy(logits, answ_id)
            relative_accuracy_results[model_id][prompt] = get_relative_accuracy(logits, answ_id)
        
        #move tensors out of gpu and delete variables (TODO also for model)
        logits = logits.detach().cpu()
        answ_id = answ_id.detach().cpu()
        
        del logits
        del answ_id
        del model
    #store all results
    all_models_all_results["logit_difference"] = logit_diff_results
    all_models_all_results["absolute_accuracy"] = accuracy_results
    all_models_all_results["relative_accuracy"] = relative_accuracy_results

    #TODO decide output directory
    with open("all_metrics_results.json", "w") as outfile:
        json.dump(all_models_all_results, outfile, indent = 2)

    #TODO make plots
    
    
    #alternative----------------------------------------------------------------------------------
    # logit_diff_results, accuracy_results, relative_accuracy_results = {}, {}, {}
    # for m in models2try:
    #     #clean memory garbage
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     model_id = models2try[m]

    #     avg_logit_difference, absolute_accuracy, relative_accuracy = get_all_results(model_id, PROMPTS, triplets)
    #     logit_diff_results[m] = avg_logit_difference
    #     accuracy_results[m] = absolute_accuracy
    #     relative_accuracy_results[m] = relative_accuracy
    # #store all results
    # all_models_all_results["logit_difference"] = logit_diff_results
    # all_models_all_results["absolute_accuracy"] = accuracy_results
    # all_models_all_results["relative_accuracy"] = relative_accuracy_results


    







