# Making Instruction Finetuning Accessible to Non-English Languages: A Case Study on Swedish

This repository contains the code, datasets, and model checkpoints for the paper [Making Instruction Finetuning Accessible to Non-English Languages: A Case Study on Swedish](https://aclanthology.org/2023.nodalida-1.62/).

## Installation and Usage Guide

This document guides you through the installation and usage of this project.

### Installation

To install all necessary packages, run the `setup.sh` script. 

```bash
./setup.sh
```

### Download Models

Models can be downloaded from a Google Drive folder using the `download_models.py` script. You can specify which model you want to download directly in the script.

```bash
pip install gdown
python download_models.py
```

### Running Evaluation and Training

Here's an example of how to run an evaluation:

```python
python run_rouge_gen.py --model_name gpt-sw3 --model_path ./models/SW3-INSTRUCT/1.3b/checkpoint-6000/ --tokenizer_path AI-Sweden-Models/gpt-sw3-1.3b --test_dataset ./data/unnatural_instructions_swe/test.jsonl --max_len 2048
```

Here's an example of how to run training:

```python
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
