#templates !!!remember to remove last word!!!
PROMPTS = {"templates0" : ["The {} the {} was in", "The {} the {} was on"],
           "templates_questions" : ["Do you know which {} the {} was in", "Do you know which {} the {} was on"],
           "templates_questions2" : ["Which {} was the {} in", "Which {} was the {} on"],
           "templates_relative_neg" : ["I don't know which {} the {} was in", "I don't know which {} the {} was on"],
           "templates_relatives_aff" : ["I saw the {} which the {} was in", "I saw the {} which the {} was on"]}

models2try = {"gpt2-small" : "openai-community/gpt2",
              "gpt2-xl": "openai-community/gpt2-xl",
              "olmo-1b": "allenai/OLMo-1B-hf",   #no tlens
              "pythia-1b": "EleutherAI/pythia-1b",
              "llama3.2-1b": "meta-llama/Llama-3.2-1B",
              "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
              "distil-deepseek" :"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" , #no tlens
              "phi1":"microsoft/phi-1_5"} 
              #"gemma3-1b" :"google/gemma-3-1b-pt", #needs the latest version of transformers, not supported in this (not in tlens)
              #"bloom-1b": "bloom-1b7",
              #"opt-1b" : "opt-1.3b",
              #"gpt-neo": "gpt-neo-1.3B"