from collections import defaultdict
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from utils.read_data import UIDataset
import evaluate
import nltk
from tqdm import tqdm


FREE_IN_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
MAX_MEMORY = f'{FREE_IN_GB-2}GB'
N_GPUS = torch.cuda.device_count()
MAX_MEMORY = {i: MAX_MEMORY for i in range(N_GPUS)}

metric = evaluate.load("rouge")


def data_collator(sample, tokenizer, max_len):
    # Encoding
    input_, output = sample
    models_input = input_ + '\noutput: '
    encoded_output = tokenizer.encode(output, add_special_tokens=False, truncation=True, max_length=max_len//2-1) + [tokenizer.eos_token_id]  
    encoded_input = tokenizer.encode(models_input, add_special_tokens=False, truncation=True, max_length=max_len//2)
    
    return torch.LongTensor([encoded_input]), torch.LongTensor([encoded_output])


def compute_metrics(preds, labels, tokenizer, model_name, checkpoint_name, test_dataset):
    print("Checkpoint name: ", checkpoint_name)
    # decode preds and labels    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence    
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    dataset_id = test_dataset.split('/')[-1].split('.')[0]

    with open('rouge_results/' + model_name + '-NI/' + dataset_id + '_generations_' + checkpoint_name + '.txt', 'w+') as f:
        f.write('\n\n*******************\n\n'.join(decoded_preds))
    with open('rouge_results/' + model_name + '-NI/' + dataset_id + '_golds.txt', 'w+') as f:
        f.write('\n\n*******************\n\n'.join(decoded_labels))

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    with open('rouge_results/' + model_name +  '-NI/' + dataset_id + '_' + checkpoint_name + '_rouge_results.txt', 'a+') as f:
        f.write(checkpoint_name + '\t' + str(result) + '\n')
    
    print(result)


def run(args):
    temp = 0.75
    beams = 4
    do_sample = False
    max_new_tokens = 512

    if args.model_name.lower() == 'gpt2':
        model = AutoModelForCausalLM.from_pretrained('distilgpt2')
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    elif args.model_name.lower() == 'gpt-sw3' or args.model_name.lower() =='opt':
        model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                        device_map="auto",
                                                        load_in_8bit=True,
                                                        max_memory=MAX_MEMORY
                                                        )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, 
                                                        use_fast=False)

    test_dataset = UIDataset(args.test_dataset)
    collator = partial(data_collator, tokenizer=tokenizer, max_len=args.max_len)

    preds, all_labels = [], []
    for sample in tqdm(test_dataset):
        input_ids, labels = collator(sample)
        input_ids = input_ids.to('cuda:0')
        generated_ids = model.generate(input_ids=input_ids, 
                                       temperature=temp,
                                       max_new_tokens=max_new_tokens,
                                       num_beams=beams,
                                       do_sample=do_sample,
                                       ) 
        
        preds.append(generated_ids[0][input_ids.shape[1]:])
        all_labels.append(labels[0])

    compute_metrics(preds, all_labels, tokenizer, args.model_name, args.model_path.strip('/').split('/')[-1], args.test_dataset)


def main(args):
    run(args)


if __name__ == "__main__":
    import argparse
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bloom', type=str, help='The name of the model')
    parser.add_argument('--model_path', default='./models/BLOOM/bloom', type=str, help='The path to the model')
    parser.add_argument('--tokenizer_path', default='./models/BLOOM/bloom', type=str, help='The path to the tokenizer')
    parser.add_argument('--test_dataset', default='./data/core_data_sv.jsonl', type=str, help='The path for the dataset file')
    parser.add_argument('--max_len', type=int, help='Model\'s max sequence length')
    
    args = parser.parse_args()
    
    main(args)