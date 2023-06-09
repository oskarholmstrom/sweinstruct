# Making Instruction Finetuning Accessible to Non-English Languages: A Case Study on Swedish

This repository contains the code, datasets, and model checkpoints for the paper [Making Instruction Finetuning Accessible to Non-English Languages: A Case Study on Swedish](https://aclanthology.org/2023.nodalida-1.62/).

## Installation and Usage Guide

This document guides you through the installation and usage of this project.

### Installation

To install all necessary packages, run the `setup.sh` script. 

```bash
./setup.sh
```

**Note:** We use Python 3.8 in this project, which is compatible with torch==1.12.1. If using other python version then you may have to change to a compatabile torch version in the setup.sh script.

### Download Models

The models are accessible through the huggingface model hub:

[oskarhol/gpt-sw3-instruct-1.3b](https://huggingface.co/oskarhol/gpt-sw3-instruct-1.3b)
[oskarhol/opt-instruct-swe-1.3b](https://huggingface.co/oskarhol/opt-instruct-swe-1.3b)


### Running Evaluation and Training

Here's an example of how to run an evaluation:

```bash
python run_rouge_gen.py --model_name gpt-sw3 --model_path oskarhol/gpt-sw3-instruct-1.3b --tokenizer_path AI-Sweden-Models/gpt-sw3-1.3b --test_dataset ./data/unnatural_instructions_swe/test.jsonl --max_len 2048
```

Here's an example of how to run training:

```bash
python train.py --model_name gpt-sw3 --model_version 1.3b --model_path AI-Sweden-Models/gpt-sw3-1.3b --train_dataset ./data/unnatural_instructions_swe/train.jsonl --test_dataset ./data/unnatural_instructions_swe/test.jsonl --out_dir ./models/SW3-INSTRUCT/test/ --max_len 2048
```

**Note:**
- You must provide the paths to where your models are located. 
- Since AI-Sweden's models are in a private repository, you may need to provide authentication towards the Hugging Face model hub. You can do this with `use_auth_token` or log in with `huggingface-cli login`:

```bash
huggingface-cli login
```

---


## Citation
```
@inproceedings{holmstrom-doostmohammadi-2023-making,
    title = "Making Instruction Finetuning Accessible to Non-{E}nglish Languages: A Case Study on {S}wedish Models",
    author = {Holmstr{\"o}m, Oskar  and
      Doostmohammadi, Ehsan},
    booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may,
    year = "2023",
    address = "T{\'o}rshavn, Faroe Islands",
    publisher = "University of Tartu Library",
    url = "https://aclanthology.org/2023.nodalida-1.62",
    pages = "634--642",
}
```
## Contact

For any questions or inquiries, please contact oskar.holmstrom@liu.se.
