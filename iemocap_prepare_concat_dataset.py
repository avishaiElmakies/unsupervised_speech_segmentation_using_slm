import argparse
import torchaudio
import os
import random
import glob
import torch
import json
from utils import get_emotion_dict
import re

pattern = r"(Ses\w+_\w+) \[\d+\.\d+-\d+\.\d+\]: (.+)"

def get_speakers_dicts(text_files):
    session_emotion_dict = {}
    for text_file in text_files:
        session_emotion_dict.update(get_emotion_dict(text_file))
    male = list(filter(lambda x: '_M' in x[0], session_emotion_dict.items()))
    female = list(filter(lambda x: '_F' in x[0], session_emotion_dict.items()))
    return male,female

def get_transcriptions_dict(transcriptions_files):
    transcriptions_dict = {}
    for file in transcriptions_files:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                match = re.match(pattern,line)
                if match is None:
                    continue
                id = match.group(1)
                text = match.group(2)
                transcriptions_dict[id] = text
    return transcriptions_dict

def get_concatenation(emotion_files,sentences_wav_path,transcription_dict,resmaple_rate):
    output_wavs = []
    output_texts = []
    output_emotions = []
    start = 0
    for file,emotion in emotion_files:
        if "script" in file:
            folder = "_".join(file.split('_')[:3])
        else:
            folder = "_".join(file.split('_')[:2])
        wav_file = os.path.join(sentences_wav_path,folder,file+'.wav')
        wav,sr = torchaudio.load(wav_file)
        if sr != resmaple_rate:
            wav = torchaudio.functional.resample(wav,sr,resmaple_rate)
        output_wavs.append(wav)
        output_texts.append(transcription_dict[file].strip())
        output_emotions.append({"emo":emotion.lower(),"start":start,"end":start+wav.shape[-1]/resmaple_rate})
        start += wav.shape[-1]/resmaple_rate
    output_wav = torch.cat(output_wavs,dim=-1)
    output_text = ' '.join(output_texts)
    return output_wav,output_text,output_emotions

def main():
    parser = argparse.ArgumentParser("helper script to prepare the iemocap dataset for our experiments")
    parser.add_argument('-i','--input_folder',type=str,help='path to the input folder',required=True)
    parser.add_argument('-o','--output_folder',type=str,help='path to the output folder',required=True)
    parser.add_argument('-s','--seed',type=int,default=None,help='seed for the random number generator')
    parser.add_argument('-max','--max_concats',type=int,default=10,help='maximum number of combinations to create for each speaker')
    parser.add_argument('-min','--min_concats',type=int,default=2,help='minimum number of combinations to create for each speaker')
    parser.add_argument('--num_files', type=int, help='number of files to sample from each speaker',required=True)
    parser.add_argument('--sample_rate', type=int,default=16000, help='resample sr')
    
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    seed = args.seed
    if seed is not None:
        random.seed(seed)
    num_files = args.num_files
    sample_rate = args.sample_rate


    json_data = {}
    for i in range(1,6):
        print(f"processing session {i}")
        session_data = {}
        session_path = os.path.join(input_folder,'Session'+str(i))
        sentences_wav_path = os.path.join(session_path,'sentences/wav')
        dialog_emo_path = os.path.join(session_path,'dialog/EmoEvaluation')
        text_files = glob.glob(os.path.join(dialog_emo_path, '*.txt'))
        transcriptions_files = glob.glob(os.path.join(session_path,'dialog/transcriptions/*.txt'))
        transcription_dicts = get_transcriptions_dict(transcriptions_files)
        emotion_male,emotion_female = get_speakers_dicts(text_files)
        output_session_path = os.path.join(output_folder,'Session'+str(i))
        os.makedirs(output_session_path,exist_ok=True)
        for j in range(num_files):
            num_seg = random.randint(args.min_concats, args.max_concats)
            files = random.sample(emotion_male, num_seg)
            wav,text,emotions = get_concatenation(files,sentences_wav_path,transcription_dicts,sample_rate)
            output_wav_path = os.path.join(output_session_path,f'Sess{i}_male_{j}.wav')
            torchaudio.save(output_wav_path,wav,sample_rate)
            session_data[f"Sess{i}_male_{j}"] = {"wav_path":os.path.abspath(output_wav_path),"text":text,"segmentation":emotions,"duration":wav.shape[-1]/sample_rate}
            files = random.sample(emotion_female, num_seg)
            wav,text,emotions = get_concatenation(files,sentences_wav_path,transcription_dicts,sample_rate)
            output_wav_path = os.path.join(output_session_path,f'Sess{i}_female_{j}.wav')
            torchaudio.save(output_wav_path,wav,sample_rate)
            session_data[f"Sess{i}_female_{j}"] = {"wav_path":os.path.abspath(output_wav_path),"text":text,"segmentation":emotions,"duration":wav.shape[-1]/sample_rate}
        json_data[f"Session{i}"] = session_data
    
    with open(os.path.join(output_folder,'data.json'),'w') as f:
        json.dump(json_data,f,indent=4)

if __name__ == "__main__":
    main()