import os
from dotenv import load_dotenv

import torch
import openai
import time

from transformers import AutoModelForCausalLM, AutoTokenizer


FREE_IN_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
MAX_MEMORY = f'{FREE_IN_GB-2}GB'
N_GPUS = torch.cuda.device_count()
MAX_MEMORY = {i: MAX_MEMORY for i in range(N_GPUS)}
        

def load_model(args): ## take in name, **kwargs for hyperparameters
    print(args.model_name)
    if args.model_name == 'gpt3':
        return GPT3(args.model_name, args.model_version, args.max_new_tokens)

    if args.tokenizer_path != '':
        return Model(args.model_name, args.model_version, args.model_path, args.tokenizer_path, args.max_new_tokens)
    
    return Model(args.model_name, args.model_version, args.model_path, args.max_new_tokens)
        


class Model():
    
    def __init__(self, name, version, path, max_new_tokens=8):
        
        self.name = name
        self.model_version = version
        self.model_path = path
        self.max_new_tokens = max_new_tokens
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, 
                                                          device_map="auto", 
                                                          load_in_8bit=True,
                                                          max_memory=MAX_MEMORY)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, 
                                                       use_fast=False)
        

    def __init__(self, name, version, path, tokenizer_path, max_new_tokens=8):
        
        self.name = name
        self.model_version = version
        self.model_path = path
        self.tokenizer_path = tokenizer_path
        self.max_new_tokens = max_new_tokens
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, 
                                                          device_map="auto", 
                                                          load_in_8bit=True,
                                                          max_memory=MAX_MEMORY)
        print(self.tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, 
                                                       use_fast=False)
        
    
    def lm(self, input_ids, labels):
        return self.model(input_ids, labels=labels)
    
    def generate(self, input): # **kwargs for hyperparameters
        
        input_ids = self.tokenizer(input, return_tensors='pt').input_ids.to('cuda')
        
        generated_ids = self.model.generate(input_ids=input_ids, 
                                            temperature=0.7,
                                            max_new_tokens=self.max_new_tokens,
                                            diversity_penalty=1.0,
                                            repetition_penalty=1.0,
                                            num_beams=4,
                                            num_beam_groups=2,
                                            early_stopping=True)    
        
        output = self.tokenizer.decode(generated_ids[0], 
                                       skip_special_tokens=True)
        
        return output
    

class GPT3():
    
    def __init__(self, name, model_version, max_tokens=128):

        assert load_dotenv()
        self.name = name
        self.model_version = model_version
        openai.api_key = os.getenv("GPT_API_KEY")
        self.max_tokens = max_tokens
        
      
    def lm(self, input):
        output = openai.Completion.create(
                engine=self.model_version,
                prompt=input,
                max_tokens=0,
                logprobs=0,
                echo=True,
            )
        return output
    
    def generate(self, input): # **kwargs for hyperparameters
        
       while True:
            try:
                output = openai.Completion.create(
                    engine=self.model_version,
                    prompt=input,
                    temperature=0.7,
                    max_tokens=self.max_tokens,
                    #stop=["Svar:", "Ord:", "."],
                    #top_p=1,
                    #frequency_penalty=2.0,
                    #presence_penalty=2.0,
                )
                return output['choices'][0]['text']
            except openai.error.RateLimitError as e:
                print("Rate limit exceeded. Retrying in 20 seconds...")
                time.sleep(20)
            except Exception as e:
                print(f"An error occurred: {e}")
                raise e
    