import json
from pathlib import Path

import evaluate
import nltk


metric = evaluate.load("rouge")

def compute_metrics(gens, golds, model_name, task_name):
    # rougeLSum expects newline after each sentence    
    gens = ["\n".join(nltk.sent_tokenize(gen.strip())) for gen in gens]
    golds = ["\n".join(nltk.sent_tokenize(gold.strip())) for gold in golds]

    result = metric.compute(predictions=gens, references=golds, use_stemmer=True)
    
    with open('rouge_results/' + model_name + 'rouge_results.txt', 'a+') as f:
        f.write(task_name + '\t' + str(result) + '\n')
    
    print(result)


def main(args):
    for f in Path(args.gold_dir).iterdir():
        gens, golds = [], []
        f = str(f).split('/')[-1]
        gf = open(args.gold_dir + f)
        if 'UI' not in args.model_name:
            try:
                pf = open(args.gens_dir + 'result_' + f)
            except:
                try:
                    pf = open(args.gens_dir + 'result_curie_' + f)
                except:
                    try:
                        pf = open(args.gens_dir + 'result_curie-std_' + f)
                    except:
                        pf = open(args.gens_dir + 'result_davinci-std_' + f)

            for gen_l, gold_l in zip(gf, pf):    
                gens.append(json.loads(gen_l)['output'])
                golds.append(json.loads(gold_l)['output'])
        else:
            for gold_l in gf:    
                gens.append(json.loads(gold_l)['model_output'])
                golds.append(json.loads(gold_l)['gold_output'])

        compute_metrics(gens, golds, args.model_name, f)


if __name__ == "__main__":
    import argparse
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bloom', type=str, help='The name of the model')
    parser.add_argument('--gold_dir', type=str, help='The path for the gold data directory')
    parser.add_argument('--gens_dir', type=str, help='The path for the generations directory')
    
    args = parser.parse_args()
    
    main(args)