from abc import ABC, abstractmethod
from .scorers import Scorer,ScorerFactory
from .speech_sentencer import SpeechSentencer,SpeechSentencerFactory
from .spans_selector import SpansSelector,SpansSelectorFactory
import torch
from typing import Tuple,List,Mapping,Any
from torch import FloatTensor
from .UnsupSeg.next_frame_classifier import NextFrameClassifier
from argparse import Namespace
import torchaudio
import os

class Segmentor(ABC,torch.nn.Module):

    @abstractmethod
    def segment_path(self, wav_path: str) -> Tuple[List[FloatTensor],int]:
        """
        segments an audio from a path
        :param wav_path: path to audio file 
        :return: a tuple containing the segments, the sampling rate
        """
        pass

    @abstractmethod
    def segment_audio(self, audio: FloatTensor, sr: int) -> Tuple[List[FloatTensor],int]:
        """
        segments a audio
        :param audio: audio tensor
        :param sr: sampling rate
        :return: a tuple containing the segments, the sampling rate
        """
        pass
        
class SegmentorFactory:

    @staticmethod
    def get_segmentor(segmentor_config:Mapping,multi_gpu_load=False,base_path:str='../models/')->Segmentor:
        """
        factrory method to get a segmentor from a config
        """
        segmentor_type = segmentor_config['type']
        if segmentor_type == 'speech_pmi':
            return SpeechPMISegmentor(segmentor_config, base_path=base_path)
        elif segmentor_type == 'equal_length':
            return EqualLengthSegmentor(segmentor_config, base_path=base_path)
        elif segmentor_type == 'next_frame':
            return NextFrameSegmentor(segmentor_config)
        elif segmentor_type == 'speaker_diarization':
            return SpeakerDiarizationSegmentor(segmentor_config)
        elif segmentor_type == 'emotion_diarization':
            return EmotionDiarizationSegmentor(segmentor_config,base_path=base_path)
        else:
            raise NotImplementedError(f'No such segmentor type: {segmentor_type}')


class SpeechPMISegmentor(Segmentor):

    def __init__(self, segmentor_config:Mapping, base_path:str='../models/'):
        """
        init method for SpeechPMISegmentor
        :param segmentor_config: segmentor config: needs to contain a sentencer config, scorer config with type 'pmi' and decider config. plus a descending flag.
        :param base_path: base path for models
        """
        super().__init__()
        sentencer_config = segmentor_config['sentencer']
        scorer_config = segmentor_config['scorer']
        sselector_config = segmentor_config['sselector']
        self.scorer:Scorer = ScorerFactory.get_scorer(scorer_config, base_path)
        self.sentercer:SpeechSentencer = SpeechSentencerFactory.get_sentencer(sentencer_config)
        self.spans_selector:SpansSelector = SpansSelectorFactory.get_span_selector(sselector_config)
        self.default_sample_rate = segmentor_config.get('default_sample_rate', 16000)
        self.device = "cpu"
        

    def segment_path(self, wav_path: str)-> Tuple[List[Mapping[str,Any]],int]:
        """
        segments a list of items
        :param wav_path: path to audio file 
        :return: a tuple containing the segments, the sampling rate
        """
        sentences,sr = self.sentercer.sentence_path(wav_path,self.default_sample_rate,self.scorer.device)
        segments = self._get_segments(sentences)
        return segments, sr

    def _get_segments(self, sentences: List[Mapping[str,Any]])->List[Mapping[str,Any]]:
        """
        segments a list of sentences
        :param sentences: list of sentences
        :return: a tuple containing the segments
        """
        audio_sentences = [sent["audio"] for sent in sentences]
        scores = self.scorer.score_consecutive(audio_sentences)
        spans = self.spans_selector.get_spans(sentences,scores)
        segments = [{"audio":torch.cat(audio_sentences[l:r],dim=-1),"start":sentences[l]["start"],"end":sentences[r-1]["end"]}
                     for l,r in zip(spans[:-1],spans[1:])]
        return segments
    
    def segment_audio(self, audio: FloatTensor, sr: int) -> Tuple[List[Mapping[str,Any]], int]:
        """
        segments a audio
        :param audio: audio tensor
        :param sr: sampling rate
        :return: a tuple containing the segments, the sampling rate
        """
        sentences = self.sentercer.sentence_audio(audio,sr)
        segments = self._get_segments(sentences)
        return segments, sr

    def forward(self, sentences: List[FloatTensor])->List[Mapping[str,Any]]:
        """
        segments a list of sentences
        :param sentences: list of sentences
        :return: a tuple containing the segments
        """
        return self._get_segments(sentences)
    
    def to(self,device):
        self.scorer.to(device)
    

class EqualLengthSegmentor(Segmentor):

    """
    EqualLengthSegmentor is a Segmentor that returns segments of equal length. will segment the audio into equal length segments (in terms of sentences)
    """

    def __init__(self, segmentor_config:Mapping,base_path:str='../models/'):
        """
        init method for EqualLengthSegmentor
        :param segmentor_config: segmentor config: needs to contain a sentencer config, scorer config with type 'pmi' and decider config. plus a descending flag.
        """
        super().__init__()
        sentencer_config = segmentor_config['sentencer']
        self.sentercer:SpeechSentencer = SpeechSentencerFactory.get_sentencer(sentencer_config)
        self.default_sample_rate = segmentor_config.get('default_sample_rate', 16000)
        sselector_config = segmentor_config['sselector']
        self.spans_selector:SpansSelector = SpansSelectorFactory.get_span_selector(sselector_config)
        

    def segment_path(self, wav_path: str)-> Tuple[List[Mapping[str,Any]],int]:
        """
        segments a list of items
        :param wav_path: path to audio file 
        :return: a tuple containing the segments, the sampling rate
        """
        sentences,sr = self.sentercer.sentence_path(wav_path,self.default_sample_rate)
        segments = self._get_segments(sentences)
        return segments, sr

    def _get_segments(self, sentences: List[Mapping[str,Any]])->List[Mapping[str,Any]]:
        """
        segments a list of sentences
        :param sentences: list of sentences
        :return: a tuple containing the segments
        """
        num_segments = self.spans_selector.decide(sentences,None)
        audio_sentences = [sent["audio"] for sent in sentences]
        if len(audio_sentences) <= num_segments:
            return sentences
        spans = list(range(0,len(audio_sentences),len(audio_sentences)//(num_segments)))
        spans.append(len(audio_sentences))
        segments = [{"audio":torch.cat(audio_sentences[l:r],dim=-1),"start":sentences[l]["start"],"end":sentences[r-1]["end"]}
                     for l,r in zip(spans[:-1],spans[1:])]
        return segments
    
    def segment_audio(self, audio: FloatTensor, sr: int) -> Tuple[List[Mapping[str,Any]], int]:
        """
        segments a audio
        :param audio: audio tensor
        :param sr: sampling rate
        :return: a tuple containing the segments, the sampling rate
        """
        sentences = self.sentercer.sentence_audio(audio,sr)
        segments = self._get_segments(sentences)
        return segments, sr
    

# uses implementation from https://github.com/felixkreuk/UnsupSeg
class NextFrameSegmentor(Segmentor):

    def __init__(self, segmentor_config) -> None:
        super().__init__()
        model_ckpt = segmentor_config['model_ckpt']
        ckpt = torch.load(model_ckpt)
        hp = Namespace(**dict(ckpt["hparams"]))

        # load weights and peak detection params
        self.model = NextFrameClassifier(hp)
        weights = ckpt["state_dict"]
        weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
        self.model.load_state_dict(weights)
        self.model.eval()
        self.num_segments = segmentor_config.get('num_segments', 10)
        self.hp = hp
        self.device = "cpu"

    @torch.no_grad()
    def segment_path(self, wav_path: str)-> Tuple[List[Mapping[str,Any]],int]:
        audio, sr = torchaudio.load(wav_path)
        return self.segment_audio(audio, sr)


    def segment_audio(self, audio: FloatTensor, sr: int) -> Tuple[List[FloatTensor],int]:
        audio = audio.to(self.device)
        preds = self.model(audio)
        preds = preds[1][0].squeeze(0)
        _, indicies = torch.sort(preds, descending=True)
        indicies = indicies[:self.num_segments-1]
        indicies = torch.sort(indicies).values.tolist()
        boundries = [0] + [(i * 160) for i in indicies] + [audio.shape[-1]]
        segments = [{"audio":audio[...,l:r],"start":l/sr ,"end":r/sr} for l,r in zip(boundries[:-1],boundries[1:])]
        return segments, sr

    def to(self,device):
        self.device = device
        self.model.to(device)
        return self
        
class SpeakerDiarizationSegmentor(Segmentor):

    def __init__(self, segmentor_config) -> None:
        super().__init__()
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(segmentor_config['pipeline_name'],use_auth_token=os.environ.get("HF_HUB_TOKEN",""))
        self.device = "cpu"
        self.segment_threshold = segmentor_config.get('segment_threshold',0.25)
        self.combine_threshold = segmentor_config.get('combine_threshold',0.5)

    def segment_path(self, wav_path: str)-> Tuple[List[Mapping[str,Any]],int]:
        audio, sr = torchaudio.load(wav_path)
        return self.segment_audio(audio, sr)
    
    def segment_audio(self, audio: FloatTensor, sr: int) -> Tuple[List[Mapping[str,Any]],int]:
        diarization = self.pipeline({"waveform": audio, "sample_rate": sr})
        segmentation = [{"start":turn.start,"end":turn.end,"speaker":speaker, "audio":audio[...,int(turn.start*sr):int(turn.end*sr)]} for  turn,_,speaker in diarization.itertracks(yield_label=True) if turn.end - turn.start > self.segment_threshold]
        segments = []
        currsegments = [segmentation[0]]
        for seg in segmentation[1:]:
            if currsegments[-1]["speaker"] == seg["speaker"] and seg["start"] - currsegments[-1]["end"] < self.combine_threshold:
                currsegments.append(seg)
            else:
                segments.append({"audio":audio[...,int(currsegments[0]["start"]*sr):int(currsegments[-1]["end"]*sr)],"start":currsegments[0]["start"],"end":currsegments[-1]["end"]})
                currsegments = [seg]
        if len(currsegments) > 0:
            segments.append({"audio":audio[...,int(currsegments[0]["start"]*sr):int(currsegments[-1]["end"]*sr)],"start":currsegments[0]["start"],"end":currsegments[-1]["end"]})
        return segments, sr

    def to(self,device):
        self.device = device
        self.pipeline = self.pipeline.to(torch.device(device))
        return self


class EmotionDiarizationSegmentor(Segmentor):

    def __init__(self, segmentor_config, base_path="../models") -> None:
        from speechbrain.inference.diarization import Speech_Emotion_Diarization
        super().__init__()
        self.classifier = Speech_Emotion_Diarization.from_hparams(source=segmentor_config["source"], savedir=base_path,overrides={"wav2vec2": {"save_path": os.path.join(base_path,"wav2vec2")}})
        self.device = "cpu"

    @torch.no_grad()
    def segment_path(self, wav_path: str)-> Tuple[List[Mapping[str,Any]],int]:
        try:
            segmentation = self.classifier.diarize_file(wav_path)[wav_path]
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                segmentation = self.classifier.diarize_file(wav_path)[wav_path]
            else:
                raise e
        audio, sr = torchaudio.load(wav_path)
        for seg in segmentation:
            seg["audio"] = audio[...,int(seg["start"]*sr):int(seg["end"]*sr)]
        return segmentation, sr
    
    def segment_audio(self, audio: FloatTensor, sr: int) -> Tuple[List[Mapping[str,Any]],int]:
        raise NotImplementedError("EmotionDiarizationSegmentor does not support segment_audio")

    def to(self,device):
        self.device = device
        self.classifier = self.classifier.to(torch.device(device))
        self.classifier.device = device
        return self

    