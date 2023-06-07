#!/bin/sh

pip install --quiet torch==1.12.1 torchvision torchaudio

pip install --quiet git+https://github.com/huggingface/transformers.git

pip install —-quiet nltk

git clone https://github.com/huggingface/evaluate
cd evaluate
pip install -e .
cd ..

pip install —-quiet sentencepiece
pip install —-quiet rouge_score
pip install --quiet bitsandbytes
pip install —-quiet accelerate
pip install —-quiet scipy
