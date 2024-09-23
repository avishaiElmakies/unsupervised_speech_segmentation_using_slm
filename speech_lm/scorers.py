from abc import ABC, abstractmethod
import torch
from typing import List, Mapping
from torch import FloatTensor
from .inference import InferenceModelFactory

class Scorer(ABC, torch.nn.Module):

    @abstractmethod
    def score_consecutive(self, x: List[FloatTensor]) -> FloatTensor:
        """
        gives a score to a list of consecutive sentences
        returns a tensor of size len(x)-1 with the score of each consecutive pair
        :param x:
        :return: score
        """
        pass

    @abstractmethod
    def to(self, device):
        """
        moves scorer to device
        :param device: device
        """
        pass


class ScorerFactory:

    @staticmethod
    def get_scorer(scorer_config:Mapping,base_path="../models") -> Scorer:
        """
        factory method to get a scorer from a config
        :param scorer_config: scorer config
        :return: scorer
        """
        scorer_type = scorer_config['type']
        if scorer_type == 'pmi':
            return SpeechPMIscorer(scorer_config,base_path=base_path)
        else:
            raise NotImplementedError(f'No such scorer type: {scorer_type}')
        

class SpeechPMIscorer(Scorer):

    def __init__(self, config,base_path="../models"):
        super().__init__()
        inferance_config = config['inference_model']
        self.inference_model = InferenceModelFactory.get_model(inferance_config,base_path=base_path)
        self.batch_size = config.get('batch_size',-1)
        self.device = "cpu"


    def _batch_log_likelihood(self, sentences: List[torch.Tensor]) -> FloatTensor:
        """
        return likelihood for a batch of audio sentences
        :param sentences: batch of sentences
        :return: likelihoods
        """
        likelihoods = []
        for i in range(len(sentences),self.batch_size):
            likelihoods.append(self.inference_model.log_likelihood(sentences[i:i+self.batch_size]))
        return torch.cat(likelihoods)
    
    @torch.no_grad()
    def score_consecutive(self, sentences: List[torch.Tensor]) -> FloatTensor:
        """
        gives a score to a list of consecutive sentences
        returns a tensor of size len(sentences)-1 with the score of each consecutive pair
        sentences: List of sentences
        return: score
        """
        assert len(sentences) > 1, "need at least two sentences to score"
        concat = [torch.cat((x, y), dim=-1) for x, y in zip(sentences[:-1], sentences[1:])]
        if self.batch_size > 0:
            sentences_log_likelihoods = self._batch_log_likelihood(sentences)
            concat_log_likelihoods = self._batch_log_likelihood(concat)
        else:
            sentences_log_likelihoods = self.inference_model.log_likelihood(sentences)
            concat_log_likelihoods = self.inference_model.log_likelihood(concat)
        log_numerator = concat_log_likelihoods
        log_denominator_x1 = sentences_log_likelihoods[:-1]
        log_denominator_x2 = sentences_log_likelihoods[1:]
        return log_numerator - (log_denominator_x1 + log_denominator_x2)
    
    def to(self, device):
        """
        moves scorer to device
        """
        self.inference_model.to(device)
        self.device = device