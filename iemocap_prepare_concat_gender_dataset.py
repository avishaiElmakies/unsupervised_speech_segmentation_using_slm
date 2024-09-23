import argparse
import torchaudio
import os
import random
import glob
import torch
import json
from utils import get_emotion_dict,EMOTION_DICT
import re

pattern = r"(Ses\w+_\w+) \[\d+\.\d+-\d+\.\d+\]: (.+)"


EMOTIONS = list(set(EMOTION_DICT.values()))

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


def get_speaker_dict(text_files):
    male_speaker_dict = {}
    female_speaker_dict = {}
    for text_file in text_files:
        emotion_dict = get_emotion_dict(text_file)
        for file,emo in emotion_dict.items():
            if '_M' in file:
                if emo not in male_speaker_dict:
                    male_speaker_dict[emo] = []
                male_speaker_dict[emo].append(file)
            elif '_F' in file:
                if emo not in female_speaker_dict:
                    female_speaker_dict[emo] = []
                female_speaker_dict[emo].append(file)
    return male_speaker_dict,female_speaker_dict

get_folder = lambda x: "_".join(x.split('_')[:3]) if "script" in x else "_".join(x.split('_')[:2])

def get_wav(file,sentences_wav_path,resmaple_rate):
    folder = get_folder(file)
    wav_file = os.path.join(sentences_wav_path,folder,file+'.wav')
    wav,sr = torchaudio.load(wav_file)
    if sr != resmaple_rate:
        wav = torchaudio.functional.resample(wav,sr,resmaple_rate)
    return wav

def speaker_segment(wav,sample_rate,speaker,start):
    return {
        "speaker":speaker,
        "start":start,
        "end":start+wav.shape[-1]/sample_rate
    }, start+wav.shape[-1]/sample_rate


def concat_files(first_files,second_files,first_speaker,second_speaker, 
                 sentences_wav_path,transcription_dict,resmaple_rate):
    output_wavs = []
    output_texts = []
    output_speakers = []
    start = 0
    for file1,file2 in zip(first_files,second_files):
        wav1 = get_wav(file1,sentences_wav_path,resmaple_rate)
        wav2 = get_wav(file2,sentences_wav_path,resmaple_rate)
        output_wavs.extend([wav1,wav2])
        output_texts.extend([transcription_dict[file1],transcription_dict[file2]])
        speaker1_seg,start = speaker_segment(wav1,resmaple_rate,first_speaker,start)
        speaker2_seg,start = speaker_segment(wav2,resmaple_rate,second_speaker,start)
        output_speakers.extend([speaker1_seg,speaker2_seg])
    output_wav = torch.cat(output_wavs,dim=-1)
    output_text = ' '.join(output_texts)
    return output_wav,output_text,output_speakers

def get_speakers_dict(input_folder):
    """
    get a dict where each key is a speaker, values are another dict with a key for each emotion that will have a list of files
    """
    male_speakers_dict = {}
    female_speakers_dict = {}
    transcription_dict = {}
    for i in range(1,6):
        session_path = os.path.join(input_folder,'Session'+str(i))
        dialog_emo_path = os.path.join(session_path,'dialog/EmoEvaluation')
        text_files = glob.glob(os.path.join(dialog_emo_path, '*.txt'))
        male_speaker_dict,female_speaker_dict = get_speaker_dict(text_files)
        male_speakers_dict[f"Session{str(i)}_M"] = male_speaker_dict
        female_speakers_dict[f"Session{str(i)}_F"] = female_speaker_dict
        transcriptions_files = glob.glob(os.path.join(session_path,'dialog/transcriptions/*.txt'))
        transcription_dict.update(get_transcriptions_dict(transcriptions_files))
    return male_speakers_dict,female_speakers_dict,transcription_dict

def main():
    parser = argparse.ArgumentParser("helper script to prepare the iemocap dataset for our experiments")
    parser.add_argument('-i','--input_folder',type=str,help='path to the input folder')
    parser.add_argument('-o','--output_folder',type=str,help='path to the output folder')
    parser.add_argument('-s','--seed',type=int,default=None,help='seed for the random number generator')
    parser.add_argument('-max','--max_concats',type=int,default=10,help='maximum number of combinations to create for each speaker')
    parser.add_argument('-min','--min_concats',type=int,default=2,help='minimum number of combinations to create for each speaker')
    parser.add_argument('--num_files', type=int, help='number of files to sample from each speaker')
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
    male_speakers_dict,female_speakers_dict,transcription_dict = get_speakers_dict(input_folder)
    for i in range(1,6):
        print(f"processing session {i}")
        sentences_wav_path = os.path.join(input_folder,'Session'+str(i),'sentences/wav')
        output_session_folder = os.path.join(output_folder,f"Session_{str(i)}")
        os.makedirs(output_session_folder,exist_ok=True)
        male_speaker = f"Session{str(i)}_M"
        female_speaker = f"Session{str(i)}_F"
        session_data = {}
        for j in range(num_files):
            rand_emo = random.choice(EMOTIONS)
            num_segments = random.randint(args.min_concats,args.max_concats)
            male_emo_list = male_speakers_dict[male_speaker][rand_emo]
            female_emo_list = female_speakers_dict[female_speaker][rand_emo]
            wav,text,speaker_segments = concat_files(random.sample(male_emo_list,num_segments),random.sample(female_emo_list,num_segments),
                                        male_speaker,female_speaker,sentences_wav_path,transcription_dict,sample_rate)
            output_wav_path = os.path.join(output_session_folder,f"MF_{rand_emo}_{j}.wav")
            torchaudio.save(output_wav_path,wav,sample_rate)
            session_data[f"MF_{rand_emo}_{j}"] = {
                "text":text,
                "segmentation":speaker_segments,
                "wav_path":os.path.abspath(output_wav_path),
                "duration":wav.shape[-1]/sample_rate,
                "emotion":rand_emo
            }

            rand_emo = random.choice(EMOTIONS)
            num_segments = random.randint(args.min_concats,args.max_concats)
            male_emo_list = male_speakers_dict[male_speaker][rand_emo]
            female_emo_list = female_speakers_dict[female_speaker][rand_emo]
            wav,text,speaker_segments = concat_files(random.sample(female_emo_list,num_segments),random.sample(male_emo_list,num_segments),
                                        female_speaker,male_speaker,sentences_wav_path,transcription_dict,sample_rate)
            output_wav_path = os.path.join(output_session_folder,f"FM_{rand_emo}_{j}.wav")
            torchaudio.save(output_wav_path,wav,sample_rate)
            session_data[f"FM_{rand_emo}_{j}"] = {
                "text":text,
                "segmentation":speaker_segments,
                "wav_path":os.path.abspath(output_wav_path),
                "duration":wav.shape[-1]/sample_rate,
                "emotion":rand_emo
            }
        json_data[f"Session_{str(i)}"] = session_data

    with open(os.path.join(output_folder,'data.json'),'w') as f:
        json.dump(json_data,f,indent=4)


if __name__ == "__main__":
    main()