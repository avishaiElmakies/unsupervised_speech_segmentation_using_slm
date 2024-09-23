import os
import re
import torch
from librosa import get_duration

VAD_SR = 16000
VAD_THRESHOLD = 0.4


def load_vad():
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False, trust_repo=True)
    return vad_model, utils


def split_audio(audio, sr, seconds):
    seg_length = int(seconds * sr)
    chunks = torch.split(audio, seg_length, dim=-1)
    return chunks


def get_folder_duration(path, folder):
    total_duration = 0
    folder_path = os.path.join(path, folder)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(root, file)
            total_duration += get_duration(path=file_path)
    return total_duration

def load_utterInfo(inputFile):
    """
    Load utterInfo from original IEMOCAP database
    """
    # this regx allow to create a list with:
    # [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
    # [V, A, D] means [Valence, Arousal, Dominance]
    pattern = re.compile(
        "[\[]*[0-9]*[.][0-9]*[ -]*[0-9]*[.][0-9]*[\]][\t][a-z0-9_]*[\t][a-z]{3}[\t][\[][0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[\]]",
        re.IGNORECASE,
    )  # noqa
    with open(inputFile, "r") as myfile:
        data = myfile.read().replace("\n", " ")
    result = pattern.findall(data)
    out = []
    for i in result:
        a = i.replace("[", "")
        b = a.replace(" - ", "\t")
        c = b.replace("]", "")
        x = c.replace(", ", "\t")
        out.append(x.split("\t"))
    return out


EMOTION_DICT = {
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "ang": "angry",
    "neu": "neutral",
}

def get_emotion_dict(text_file):
    """
    Get emotion dict from original IEMOCAP database
    """
    emotion_dict = {}
    with open(text_file) as f:
        utterance = load_utterInfo(text_file)
        for line in utterance:
            id = line[2]
            emo = line[3]
            if emo in EMOTION_DICT:
                emotion_dict[id] = EMOTION_DICT[emo]
    return emotion_dict