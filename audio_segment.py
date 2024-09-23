import argparse
import json
from tqdm import tqdm
from speech_lm.segmentors import Segmentor, SegmentorFactory
import torchaudio
import os
import torch

def main():
    parser = argparse.ArgumentParser("this script segments auido files into smaller chunks.")
    parser.add_argument('-c','--seg_config', type=str, help='Path to the config file for segmentation',required=True)
    parser.add_argument('-j','--input_json', type=str, help='Path to the json file that contains the wav paths. The keys are the names of the folders. each key will have multiple sub keys (sub folders). with a wav path as the value.',required=True)
    parser.add_argument('-o','--output_folder', type=str, help='folder to save the segmented audio files',required=True)
    parser.add_argument('-b','--base_path', type=str, default='../models/', help='base path for models')
    parser.add_argument("-s",'--save_params',action='store_true',help="save the params and data json to reproduce the results if needed")
    parser.add_argument('-sa','--save_audio',action='store_true',help='bool do decide if the audio segments should be saved or not')

    args = parser.parse_args()


    input_json = args.input_json
    output_folder = args.output_folder
    save_audio = args.save_audio

    with open(args.seg_config) as f:
        seg_config = json.load(f)

    segmentor:Segmentor = SegmentorFactory.get_segmentor(seg_config,base_path=args.base_path)
    segmentor.eval()
    if torch.cuda.is_available():
        segmentor.to('cuda')
        segmentor.eval()

    with open(input_json) as f:
        data = json.load(f)
    
    os.makedirs(output_folder,exist_ok=True)

    json_segments = {}
    for key, value in tqdm(data.items(),total=len(data)):
        json_segments[key] = {}
        for sub_key, sub_values in tqdm(value.items(),total=len(value)):
            if isinstance(sub_values,str):
                wav_path = sub_values
            elif isinstance(sub_values,dict):
                wav_path = sub_values["wav_path"]
            try:
                segments, sr = segmentor.segment_path(wav_path)
            except RuntimeError as e:
                print(f"Error in {key}/{sub_key}: {e}")
                continue
            wavs = [segment.pop("audio") for segment in segments]
            if wavs[0].ndim > 1:
                wavs = [wav.mean(dim=0) for wav in wavs]
            json_segments[key][sub_key] = {}
            if save_audio:
                outdir = os.path.join(output_folder,key,sub_key)
                os.makedirs(outdir,exist_ok=True)
                for i, (wav,segment) in enumerate(zip(wavs,segments)):
                    outpath = os.path.join(outdir,f'{sub_key}_{i}.wav')
                    torchaudio.save(outpath,wav.unsqueeze(0),sr)
                    segment["audio_path"] = os.path.abspath(outpath)
                json_segments[key][sub_key]["audio_folder"] = os.path.abspath(outdir)
            json_segments[key][sub_key]["segmentation"] = segments
                

    with open(os.path.join(output_folder,'results.json'),'w') as f:
        json.dump(json_segments,f,indent=4)

    if args.save_params:
        with open(os.path.join(output_folder,'seg_config.json'),'w') as f:
            json.dump(seg_config,f,indent=4)
        with open(os.path.join(output_folder,'data.json'),'w') as f:
            json.dump(data,f,indent=4)

if __name__ == '__main__':
    main()