from abc import ABC, abstractmethod
from torch import Tensor
import torchaudio
from typing import List, Tuple,Mapping,Any
import torch


class SpeechSentencer(ABC):
    """
    SpeechSentencer is an abstract class that creates "sentences" from audio.
    for an audio file. it will create x1, x2, x3, ... xn "sentences" for the
    audio file. The sentences are not necessarily sentences in the traditional
    sense, but they are segments of the audio file that are related in some
    way.
    """

    @abstractmethod
    def sentence(self, audio: Tensor, sr: int) -> List[Mapping[str,Any]]:
        """
        Sentence audio
        :param sr: sampling rate of the audio
        :param audio: audio to sentence
        :return: list of audio sentences
        """
        pass

    @abstractmethod
    def sentence_path(self, audio_file: str, resample_rate:int = -1, device="cpu") -> Tuple[List[Mapping[str,Any]], int]:
        """
        Sentence audio file
        :param audio_file: audio file to sentence
        :param resample_rate: resample rate. if -1, no resampling will be done if different from audio sr will resample.
        :return: list of audio sentences
        """
        pass

class SpeechSentencerFactory:

    @staticmethod
    def get_sentencer(sentencer_config:Mapping)->SpeechSentencer:
        """
        factrory method to get a sentencer from a config
        """
        sentencer_type = sentencer_config['type']
        if sentencer_type == 'length':
            return LengthSpeechSentencer(sentencer_config['length'],sentencer_config["min_length"],sentencer_config['drop_last'])
        else:
            raise NotImplementedError(f'No such sentencer type: {sentencer_type}')


class LengthSpeechSentencer(SpeechSentencer):
    """
    LengthSpeechSentencer is a SpeechSentencer that uses the length of the
    audio to sentence it. It will divide the audio into x1, x2, x3, ... xn
    "sentences" of equal audio length.
    """

    def __init__(self, length: int,min_length=0.05,drop_last=False):
        """
        :param length: length of each sentence in seconds
        :param min_length: check if the last sentence is shorter than min length. 
                           if he is combine the last two sentences
        :param drop_last: drop last sentece
        """
        self.length = length
        self.drop_last = drop_last
        self.min_length = min_length
        

    def sentence(self, audio: Tensor, sr: int) -> List[Mapping[str,Any]]:
        """
        :param audio: audio to sentence
        :param sr: sampling rate of the audio
        :return: list of audio sentences with their start and end times
        """
        l = int(self.length * sr)
        sentences = [
                     {"audio":audio[..., i:i + l], "start":i/sr, "end":min((i + l)/sr,audio.shape[-1]/sr)}
                     for i in range(0, audio.shape[-1], l)
                    ]
        if self.drop_last:
            sentences.pop()
        if sentences[-1]["audio"].shape[-1] < int(self.min_length * sr):
            last = sentences.pop()
            prev_last = sentences.pop()
            sentences.append(
                    {"audio":torch.concat((prev_last["audio"],last["audio"]),dim=-1),"start":prev_last["start"],"end":last["end"]}
                 )
        return sentences


    def sentence_path(self, audio_path: str, resample_rate:int = -1,device="cpu") -> Tuple[List[Mapping[str,Any]], int]:
        """
        Sentence audio file
        :param audio_path: audio file path to sentence
        :return: tuple of list of audio sentences and sampling rate
        """
        audio, sr = torchaudio.load(audio_path)
        audio = audio.to(device)
        if resample_rate != -1 and sr != resample_rate:
            audio = torchaudio.functional.resample(audio,sr,resample_rate)
            sr = resample_rate
        return self.sentence(audio, sr), sr
