import argparse
import torchaudio
import os
import random
import glob
import torch
import json
from utils import load_vad,VAD_THRESHOLD

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

    resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=args.sample_rate)

    vad_model, utils = None,None
    get_speech_timestamps, _, _, _,collect_chunks = None,None,None,None,None
    if args.remove_silence:
        vad_model, utils = load_vad()
        get_speech_timestamps, _, _, _,collect_chunks = utils

    json_data = {}
    for i in range(1,5):
        speaker_data = {}
        speaker_input_folder = os.path.join(input_folder,str(i))
        speaker_output_folder = os.path.join(output_folder,str(i))
        os.makedirs(speaker_output_folder,exist_ok=True)
        all_files = glob.glob(os.path.join(speaker_input_folder, '*.wav'))
        if remove_emotions is not None:
            all_files = [f for f in all_files if not any([emotion in f for emotion in remove_emotions])]
        for j in range(num_files):
            num_seg = random.randint(args.min_concats, args.max_concats)
            files = random.sample(all_files, num_seg)
            output_wavs = []
            output_texts = []
            output_emotions = []
            start = 0
            for file in files:
                wav,sr = torchaudio.load(file)
                if sr != args.sample_rate:
                    wav = resample(wav)
                if vad_model is not None:
                    timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sr,threshold=VAD_THRESHOLD)
                    if len(timestamps) > 0:
                        wav = collect_chunks(timestamps,wav.squeeze(0)).unsqueeze(0)
                output_wavs.append(wav) 
                lab_file = file.replace('.wav','.lab')
                with open(lab_file) as f:
                    text = f.read()
                output_texts.append(text)
                emotion = os.path.basename(lab_file).split('_')[0]
                output_emotions.append({"emo":emotion.lower(),"start":start,"end":start+wav.shape[-1]/args.sample_rate})
                start += wav.shape[-1]/args.sample_rate
            output_wav = torch.cat(output_wavs,dim=-1)
            output_text = '.'.join(output_texts)
            output_wav_path = os.path.join(speaker_output_folder,f'{j}.wav')
            torchaudio.save(output_wav_path,output_wav,args.sample_rate)
            speaker_data[f"{j}"] = {"wav_path":os.path.abspath(output_wav_path),"text":output_text,"segmentation":output_emotions,"duration":output_wav.shape[-1]/args.sample_rate}
        json_data[f"speaker_{i}"] = speaker_data

    with open(os.path.join(output_folder,'data.json'),'w') as f:
        json.dump(json_data,f,indent=4)
            
if __name__ == "__main__":
    main()