pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/pytorch/fairseq@da8fb630880d529ab47e53381c30ddc8ad235216
git clone git@github.com:facebookresearch/textlesslib.git
cd textlesslib
pip install -e .