import torch
import numpy as np
import transformers
from transformers import pipeline
from transformers import T5Tokenizer, AutoModelForCausalLM

transformers.set_seed(7)
generator = pipeline('text-generation',model='mymodel/')
# generator = pipeline('text-generation', model='rinna/japanese-gpt2-medium')

def saying_gen(saying):
    return generator(saying, 
        max_length = len(saying)+10,
        num_return_sequences=1,      
        do_sample = True,
        num_beams=5, 
        no_repeat_ngram_size=2, 
        top_k=0,
        temperature=0.9)


saying_gen('こんにちは')
saying_gen('あなたは占い師ですか')

