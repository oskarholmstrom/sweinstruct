#!/bin/sh

conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install --quiet bitsandbytes
pip install --quiet git+https://github.com/huggingface/transformers.git # Install latest version of transformers
pip install --quiet accelerate

