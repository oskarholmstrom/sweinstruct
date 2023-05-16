from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.read_data import UIDataset
import evaluate
import nltk


FREE_IN_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
MAX_MEMORY = f'{FREE_IN_GB-2}GB'
N_GPUS = torch.cuda.device_count()
MAX_MEMORY = {i: MAX_MEMORY for i in range(N_GPUS)}

#transformers.logging.set_verbosity_info()


class Train:
    def __init__(self, args):
        if args.model_name.lower() == 'gpt2':
            self.model = AutoModelForCausalLM.from_pretrained('distilgpt2')
            self.tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        elif args.model_name.lower() == 'gpt-sw3' or args.model_name.lower() =='opt':
            self.model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                            device_map="auto",
                                                            use_cache=False)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, 
                                                            use_fast=False)

        self.train_dataset = UIDataset(args.train_dataset)
        self.test_dataset = UIDataset(args.test_dataset)
        self.max_len = args.max_len
        self.resume_from_checkpoint = args.resume_from_checkpoint

        metric = evaluate.load("rouge")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds

            # decode preds and labels    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # rougeLSum expects newline after each sentence    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            return result

        training_args = TrainingArguments(
            output_dir=args.out_dir,
            evaluation_strategy="no",
            #eval_steps=1000,
            num_train_epochs=args.num_train_epochs,
            learning_rate=2e-5,
            weight_decay=0.01,
            push_to_hub=False,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            #use_cache=False,
            eval_accumulation_steps=1,
            warmup_ratio=0.1,
            no_cuda=False,
            ddp_find_unused_parameters=False,
            local_rank=args.local_rank,
            deepspeed=args.deepspeed,
            #sharded_ddp='simple'
            #report_to="none",
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=self.data_collator,
            #compute_metrics=compute_metrics,
        )

    def data_collator(self, batch):
        # Encoding
        encoded_batch = defaultdict(list)
        for input_, output in batch:
            models_input = input_ + '\noutput: '
            encoded_output = self.tokenizer.encode(output, add_special_tokens=False, truncation=True, max_length=self.max_len//2-1) + [self.tokenizer.eos_token_id]  
            encoded_input = self.tokenizer.encode(models_input, add_special_tokens=False, truncation=True, max_length=self.max_len//2)

            encoded_batch['input_ids'].append(encoded_input + encoded_output)
            encoded_batch['labels'].append([-100 for _ in range(len(encoded_input))] + encoded_output)
            encoded_batch['attention_mask'].append([1 for _ in range(len(encoded_input))] + [0 for _ in range(len(encoded_output))])

        # Padding
        encoded_batch['input_ids'] = pad_sequence([torch.LongTensor(x) for x in encoded_batch['input_ids']], batch_first=True) 
        encoded_batch['labels'] = pad_sequence([torch.LongTensor(x) for x in encoded_batch['labels']], batch_first=True) 
        encoded_batch['attention_mask'] = pad_sequence([torch.LongTensor(x) for x in encoded_batch['attention_mask']], batch_first=True) 
        
        return dict(encoded_batch)
    
    def train(self):
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)


def main(args):
    trainer = Train(args) 
    trainer.train()


if __name__ == "__main__":
    import argparse
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bloom', type=str, help='The name of the model')
    parser.add_argument('--model_version', default='', type=str, help='The version of the model')
    parser.add_argument('--model_path', default='./models/BLOOM/bloom', type=str, help='The path to the model')
    parser.add_argument('--tokenizer_path', default='', type=str, help='The path to the tokenizer')
    parser.add_argument('--train_dataset', default='./data/core_data_sv.jsonl', type=str, help='The path for the dataset file')
    parser.add_argument('--test_dataset', default='./data/core_data_sv.jsonl', type=str, help='The path for the dataset file')
    parser.add_argument('--out_dir', type=str, help='The path to save the model in')
    parser.add_argument('--num_train_epochs', type=int, default=6, help='Number of epochs for training')
    parser.add_argument('--max_len', type=int, help='Model\'s max sequence length')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    
    args = parser.parse_args()
    
    main(args)