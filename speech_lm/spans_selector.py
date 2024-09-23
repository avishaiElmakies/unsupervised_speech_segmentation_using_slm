from abc import ABC, abstractmethod
from torch import FloatTensor
from typing import List,Mapping
import torch


class SpansSelector(ABC):
    """
    SpansSelector is an abstract class that gets the sentences and scores and decides the spans for the segmentations
    """

    @abstractmethod
    def decide(self,sentences: List[FloatTensor], scores: FloatTensor) -> int:
        """
        decides the number of segments to use
        :param sentences: list of sentences
        :param scores: scores neighbor sentences
        :return: number of segments
        """
        pass

    @abstractmethod
    def get_spans(self,sentences: List[FloatTensor], scores: FloatTensor) -> List[int]:

        """
        gets the spans for the segmentations
        :param sentences: list of sentences
        :param scores: scores neighbor sentences
        :return: list of spans
        """
        pass

class SpansSelectorFactory:

    @classmethod
    def get_span_selector(cls, sselector_config:Mapping)->SpansSelector:
        """
        factrory method to get a decider from a config
        """
        sselector_type = sselector_config['type']
        if sselector_type == 'constant':
            return ConstantSpansSelector(sselector_config)
        elif sselector_type == 'adaptive':
            return AdaptiveSpansSelector(sselector_config)
        elif sselector_type == 'threshold':
            return ThresholdSpansSelector(sselector_config)
        else:
            raise NotImplementedError(f'No such decider type: {sselector_type}')


class ConstantSpansSelector(SpansSelector):
    """
    ConstantSpansSelector is a SpansSelector that always returns the same number of segments.
    """

    def __init__(self, config: Mapping):
        """
        :param config: config for the decider. needs to contain a num_segments int.
        """
        self.num_segments = config["num_segments"]
        self.descending = config.get("descending", False)

    def decide(self, _: List[FloatTensor], __: FloatTensor) -> int:
        """
        decides the number of segments to use
        :param sentences: list of sentences
        :return: number of segments
        """
        return self.num_segments
    
    def get_spans(self, sentences: List[FloatTensor], scores: FloatTensor) -> List[int]:
        """
        gets the spans for the segmentations
        :param sentences: list of sentences
        :param scores: scores neighbor sentences
        :return: list of spans
        """
        scores, indicies = torch.sort(scores, descending=self.descending)
        top_indicies = indicies[:self.num_segments - 1]
        argsort = torch.argsort(top_indicies)
        spans = [0] + (top_indicies[argsort] + 1).detach().cpu().tolist()
        spans.append(len(sentences))
        return spans
    
class AdaptiveSpansSelector(SpansSelector):
    """
    AdaptiveSpansSelector is a SpansSelector that returns an Adaptive number of segments based on the number of sentences.
    """

    def __init__(self, config: Mapping):
        """
        :param config: config for the decider. needs to contain a base int, len_offset and sentences_for_segment int.
        """
        
        self.base_segments = config["base_segments"]
        self.len_offset = config["len_offset"]
        self.sentences_for_segment = config["sentences_for_segment"]
        self.descending = config.get("descending", False)

    def decide(self, sentences: List[FloatTensor], _: FloatTensor) -> int:
        """
        decides the number of segments to use
        :param sentences: list of sentences
        :return: number of segments
        """
        return max(0,len(sentences) - self.len_offset) // self.sentences_for_segment + self.base_segments
    
    def get_spans(self, sentences: List[FloatTensor], scores: FloatTensor) -> List[int]:
        num_segments = self.decide(sentences, scores)
        scores, indicies = torch.sort(scores, descending=self.descending)
        top_indicies = indicies[:num_segments - 1]
        argsort = torch.argsort(top_indicies)
        spans = [0] + (top_indicies[argsort] + 1).detach().cpu().tolist()
        spans.append(len(sentences))
        return spans


class ThresholdSpansSelector(SpansSelector):
    """
    ThresholdSpansSelector is a SpansSelector that returns an Adaptive number of segments based on the number of sentences.
    """

    def __init__(self, config: Mapping):
        """
        :param config: config for the decider. needs to contain a base int, len_offset and sentences_for_segment int.
        """
        self.threshold = config["threshold"]
        self.larger_than = config.get("larger_than", False)

    def decide(self, _: List[FloatTensor], scores: FloatTensor) -> int:
        """
        decides the number of segments to use
        :param sentences: list of sentences
        :return: number of segments
        """
        return self.threshold_items(scores).sum().item()
        
    def threshold_items(self, scores: FloatTensor) -> FloatTensor:
        if self.larger_than:
            return scores > self.threshold
        else:
            return scores < self.threshold
    
    def get_spans(self, sentences: List[FloatTensor], scores: FloatTensor) -> List[int]:
        indicies = self.threshold_items(scores).nonzero().squeeze(1)
        spans = [0] + (indicies + 1).detach().cpu().tolist()
        spans.append(len(sentences))
        return spans