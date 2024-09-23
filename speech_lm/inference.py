from abc import ABC, abstractmethod
import torch
from typing import List, Mapping
from .tokenizers import SpeechTokenizer
from .utils import build_speech_lm,nll
from torch.nn.utils.rnn import pad_sequence

class InferenceModel(ABC):
    
    @abstractmethod
    def log_likelihood(self, wavs: List[torch.Tensor]) -> torch.Tensor:
        ...

    @abstractmethod
    def to(self, device):
        ...


class InferenceModelFactory:

    @staticmethod
    def get_model(config: Mapping,base_path="./") -> InferenceModel:
        if config["model_type"] == "slm":
            return SLMInferenceModel(config,base_path=base_path)
        
        raise ValueError(f"Model type {config['model_type']} not supported")
    

class SLMInferenceModel(InferenceModel):

    def __init__(self, config,base_path="./"):
        tokenizer_config = config['tokenizer']
        self.tokenizer = SpeechTokenizer(tokenizer_config)
        self.speech_lm = build_speech_lm(config["model_name"], base_path=base_path)
        self.mean_nll = config.get("mean_nll",False)
        self.offset = self.speech_lm.config.offset
        self.padding_value = self.speech_lm.config.pad_token_id

    def log_likelihood(self, wavs: List[torch.Tensor]) -> torch.Tensor:
        sentece_tokens = self.tokenizer(wavs,self.offset)
        x = pad_sequence(sentece_tokens,batch_first=True,padding_value=self.padding_value)
        logits = self.speech_lm(input_ids=x).logits
        shifted_x = x[..., 1:]
        shifted_logits = logits[..., :-1, :]

        # Create a mask that is True where the tokens are not padding tokens
        mask = (shifted_x != self.padding_value)

        # Convert the losses to likelihoods

        return -nll(shifted_logits, shifted_x, mask,self.mean_nll)


    def to(self, device):
        self.tokenizer.to(device)
        self.speech_lm.to(device)
        return self
