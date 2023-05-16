
import sys
import argparse
from datetime import datetime


import json
import torch
from tqdm import tqdm
from models import load_model
import json

def read_svt_news(path):
    with open(path, 'r') as f:
        data = json.load(f)
        input = []
        for d in data:
            input.append(d['body'])
        return input


def main(args):
    now = datetime.now()
    output_file = 'results/perplexity/' + args.model_name + '_' + args.model_version + '_' + args.dataset.replace('/','-') + '_' + now.strftime("%Y-%m-%d_%H:%M") + '.txt'
    
    f = open(output_file, "w")
    f.write(f"Args: {args}\n\n\n")
    print("Args: {args}\n\n\n")
    print("Output file: ", output_file)
        
    model = load_model(args)
    
    max_length = args.max_length
    
    
    print("max length: ", max_length)
    doc_ppl = []

    dataset = read_svt_news(args.dataset)
    for i, input in enumerate(dataset):
        print(f"Doc: {i} / {len(dataset)}")

        if args.model_name != 'gpt3':
            encodings = model.tokenizer(input, return_tensors="pt")
            seq_len = encodings.input_ids.size(1)
        else:
            seq_len = len(input.split())
            
        char_in_input = len(input)

        nlls = []
        prev_end_loc = 0
        
        for begin_loc in tqdm(range(0, seq_len, args.stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            if args.model_name != 'gpt3':
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda:0')
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                if args.model_name == 'gpt3':
                    outputs = model.lm(input)
                    logprobs = outputs['choices'][0]['logprobs']['token_logprobs']
                    logprobs[0] = 0.0
                    neg_log_likelihood = -1*torch.tensor(logprobs).sum()
                else:
                    outputs = model.lm(input_ids, labels=target_ids)
                    print(outputs.loss)
                    neg_log_likelihood = outputs.loss * trg_len
                    

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
            
            assert end_loc != seq_len, 'Issue with seq_len > max_length'
        
        ppl = torch.exp(torch.stack(nlls).sum() / char_in_input)
        doc_ppl.append(ppl.item())
        f.write(f"Document: {i}  Perplexity: {ppl.item()}\n")
        print(f"Document: {i}  Perplexity: {ppl.item()}\n")

    
    final_ppl = sum(doc_ppl) / len(doc_ppl)
    f.write(f"Perplexity over corpus: {final_ppl}\n")
    print(f"Perplexity over corpus: {final_ppl}\n")
    f.close()

if __name__ == "__main__":
    import argparse
    
    print("Running perplexity")    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', default='gpt3', type=str, help='The name of the model')
    parser.add_argument('--model_version', default='text-davinci-003', type=str, help='The version of the model')
    parser.add_argument('--model_path', default='./models/GPT-SW3/gpt-sw3-1.3b', type=str, help='The path to the model')
    parser.add_argument('--tokenizer_path', default='', type=str, help='The path to the model')
    parser.add_argument('--max_length', default=2048, type=int, help='Max sequence length for the model')
    parser.add_argument('--max_new_tokens', default=512, type=int, help='Max sequence length for the model')
    parser.add_argument('--dataset', default='./data/svt_news/svt_data.json', type=str, help='The path for the dataset file')
    #parser.add_argument('--max_seq_length', default=2048, type=int, help='Maximum sequence length')
    parser.add_argument('--stride', default=1, type=int, help='Stride length for the sliding window')
    parser.add_argument('--char_norm', default=True, type=bool, help='Whether to normalize the text with bits per characters')
    

    
    args = parser.parse_args()
    
    main(args)
