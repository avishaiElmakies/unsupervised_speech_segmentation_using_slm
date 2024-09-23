import argparse
import torchaudio
import os
import random
import glob
import torch
import json
from utils import load_vad,VAD_THRESHOLD


def get_speaker_dict(text_files):
    emotion_dict = {}
    emotions = set()
    for text_file in text_files:
        with open(text_file) as f:
            text = f.read()
        emotion = os.path.basename(text_file).split('_')[0].lower()
        if emotion not in emotion_dict:
            emotion_dict[emotion] = []
        emotions.add(emotion)
        wav_file = text_file.replace('.lab','.wav')
        emotion_dict[emotion].append((wav_file,text))
    return emotion_dict

def get_speaker_list(input_folder,remove_emotions=None):

    speaker_list = []
    for i in range(1,5):
        speaker_input_folder = os.path.join(input_folder,str(i))
        all_files = glob.glob(os.path.join(speaker_input_folder, '*.lab'))
        if remove_emotions is not None:
            all_files = [f for f in all_files if not any([emotion in f for emotion in remove_emotions])]
        speaker_list.append(get_speaker_dict(all_files))
    return speaker_list

FEMALE_IDX = [0,1]
MALE_IDX = [2,3]

def get_rand_speaker(is_male):
    return random.choice(MALE_IDX if is_male else FEMALE_IDX)

def get_wav_file(wav_file,sample_rate, vad_model=None, get_speech_timestamps=None, collect_chunks=None):
    wav,sr = torchaudio.load(wav_file)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav,sr,sample_rate)
    if vad_model is not None:
        timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sample_rate,threshold=VAD_THRESHOLD)
        if len(timestamps) > 0:
            wav = collect_chunks(timestamps,wav.squeeze(0)).unsqueeze(0)
    return wav

def get_segment(speaker,start,wav,sample_rate):
    return {"speaker":speaker,"start":start,"end":start+wav.shape[-1]/sample_rate},start+wav.shape[-1]/sample_rate


def concat_files(first_list,second_list,first_speaker_idx,
                second_speaker_idx,sample_rate,vad_model=None, get_speech_timestamps=None, collect_chunks=None):
    output_wavs = []
    output_texts = []
    output_segments = []
    start = 0
    for (wav1_file,text1),(wav2_file,text2) in zip(first_list,second_list):
        wav1 = get_wav_file(wav1_file,sample_rate,vad_model, get_speech_timestamps, collect_chunks)
        wav2 = get_wav_file(wav2_file,sample_rate,vad_model, get_speech_timestamps, collect_chunks)
        output_wavs.extend([wav1,wav2])
        output_texts.extend([text1,text2])
        segment1,start = get_segment(first_speaker_idx,start,wav1,sample_rate)
        segment2,start = get_segment(second_speaker_idx,start,wav2,sample_rate)
        output_segments.extend([segment1,segment2])

    output_wav = torch.cat(output_wavs,dim=-1)
    output_text = '.'.join(output_texts)
    return output_wav,output_text,output_segments


def main():
    parser = argparse.ArgumentParser("helper script to prepare the EmoV-DB dataset for our experiments")
    parser.add_argument('-i','--input_folder',type=str,help='path to the input folder')
    parser.add_argument('-o','--output_folder',type=str,help='path to the output folder')
    parser.add_argument('-s','--seed',type=int,default=None,help='seed for the random number generator')
    parser.add_argument('-max','--max_concats',type=int,default=10,help='maximum number of combinations to create for each speaker')
    parser.add_argument('-min','--min_concats',type=int,default=2,help='minimum number of combinations to create for each speaker')
    parser.add_argument('--remove_emotions', default=None,  nargs='*', help='list of emotions to remove')
    parser.add_argument('--num_files', type=int, help='number of files to sample from each speaker')
    parser.add_argument('--sample_rate', type=int,default=16000, help='resample sr')
    parser.add_argument('--remove_silence', action='store_true', help='remove silence from the audio files')

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    seed = args.seed
    if seed is not None:
        random.seed(seed)
    remove_emotions = args.remove_emotions
    num_files = args.num_files

    vad_model, utils = None,None
    get_speech_timestamps, _, _, _,collect_chunks = None,None,None,None,None
    if args.remove_silence:
        vad_model, utils = load_vad()
        get_speech_timestamps, _, _, _,collect_chunks = utils

    speakers_list = get_speaker_list(input_folder,remove_emotions)

    json_data = {}
    for first_speaker_idx in range(len(speakers_list)):
        speaker_data = {}
        first_speaker = speakers_list[first_speaker_idx]
        male_first = first_speaker_idx in MALE_IDX
        output_speaker_folder = os.path.join(output_folder,str(first_speaker_idx+1))
        os.makedirs(output_speaker_folder,exist_ok=True)
        for second_speaker_idx in (FEMALE_IDX if male_first else MALE_IDX):
            second_speaker = speakers_list[second_speaker_idx]
            emotions = list(set(first_speaker.keys()).intersection(second_speaker.keys()))
            subkey = f"{'M' if male_first else 'F'}{first_speaker_idx+1}_{'F' if male_first else 'M'}{second_speaker_idx+1}"
            for j in range(num_files):
                emotion = random.choice(emotions)
                first_speaker_emotion_list = first_speaker[emotion]
                second_speaker_emotion_list = second_speaker[emotion]
                num_segs = random.randint(args.min_concats,args.max_concats)
                first_speaker_emotion_list = random.sample(first_speaker_emotion_list,num_segs)
                second_speaker_emotion_list = random.sample(second_speaker_emotion_list,num_segs)
                output_wav,output_text,output_segments = concat_files(first_speaker_emotion_list,second_speaker_emotion_list,
                                                                      first_speaker_idx,second_speaker_idx,args.sample_rate,
                                                                      vad_model, get_speech_timestamps, collect_chunks)
                output_wav_file = os.path.join(output_speaker_folder,f"{subkey}_{j}.wav")
                torchaudio.save(output_wav_file,output_wav,args.sample_rate)
                speaker_data[f"{subkey}_{j}"] = {"wav_path":os.path.abspath(output_wav_file),
                                        "text":output_text,
                                        "segmentation":output_segments,
                                        "emotion":emotion,
                                        "duration":output_wav.shape[-1]/args.sample_rate}
                
        json_data[f"Speaker{first_speaker_idx+1}"] = speaker_data
    

    with open(os.path.join(output_folder,'data.json'),'w') as f:
        json.dump(json_data,f,indent=4)

if __name__ == "__main__":
    main()