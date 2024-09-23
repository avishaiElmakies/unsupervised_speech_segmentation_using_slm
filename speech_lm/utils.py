import os
import zipfile
import wget
import torch
from transformers import AutoModelForCausalLM
from textless.data.speech_encoder import SpeechEncoder
from torch import FloatTensor, LongTensor
from torch.nn.functional import cross_entropy


"""
This file contains utils for speech_lm
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_URL = 'https://dl.fbaipublicfiles.com/textless_nlp/twist/lms/'


def get_gslm_speech_encoder(dense_model_name, quantizer_model_name, vocab_size,
                         deduplicate,need_f0,f0_func="yaapt",f0_normalizer=None,f0_quantizer=None,chunk_alignment=False):
    """
    get speech encoder using textless library
    :param dense_model_name: dense model name
    :param quantizer_model_name: quantizer model name
    :param vocab_size: vocab size
    :param deduplicate: deduplicate
    :param need_f0: need f0
    """
    return SpeechEncoder.by_name(
        dense_model_name=dense_model_name,
        quantizer_model_name=quantizer_model_name,
        vocab_size=vocab_size,
        deduplicate=deduplicate,
        need_f0=need_f0,
        f0_normalizer=f0_normalizer,
        f0_quantizer=f0_quantizer,
        f0_func=f0_func,
        chunk_alignment=chunk_alignment
    )

def unzip_file(zip_path, extract_path):
    """
    unzip file
    :param zip_path: path to zip file
    :param extract_path: path to extract to
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"File extracted to {extract_path}")


def maybe_download_speech_lm(name, base_path):
    """
    downloads speech lm
    :param name: name of model
    :param base_path: base path to download to
    """
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    ckpt_dir = os.path.join(base_path, name)
    if not os.path.exists(ckpt_dir):
        url = ROOT_URL + name + '.zip'
        zip_path = ckpt_dir + '.zip'
        print(f"Downloading from {url}")
        filename = wget.download(url, zip_path)
        unzip_file(filename, ckpt_dir)

    return os.path.abspath(ckpt_dir)

def build_speech_lm(model_type, base_path='./'):
    """
    builds speech lm
    retruns model
    """
    ckpt_dir = maybe_download_speech_lm(model_type, base_path)

    lm_model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
    lm_model.eval()

    return lm_model

def nll(logits:FloatTensor, target:LongTensor, mask:LongTensor,mean_nll:bool = False)->FloatTensor:
    """
    calculate the negative log likelihood of the logits given the target
    :param logits: logits
    :param target: target
    :param mask: mask
    :return: nll
    """
    # Calculate the cross-entropy loss for each sequence
    losses = cross_entropy(
        logits.contiguous().view(-1, logits.size(-1)),
        target.long().contiguous().view(-1), reduction='none')

    # Reshape the losses to match the original sequences
    losses = losses.view(*target.size())

    # Use the mask to ignore the losses of the padding tokens
    masked_losses = losses * mask

    # Sum the losses to get the total loss for each sequence
    ll = masked_losses.sum(dim=-1)
    if mean_nll:
        return ll / mask.sum(dim=-1)
    return ll