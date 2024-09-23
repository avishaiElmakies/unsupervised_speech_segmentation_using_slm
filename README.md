##  Unsupervised Speech Segmentation: A General Approach Using Speech Language Models
#### This repository is the official implementation for the paper "Unsupervised Speech Segmentation: A General Approach Using Speech Language Models" (TODO add link)

## Installing dependecies

to use the code please download the libraries needed to run it

you can run the script called ```install_dep.sh```. it will try and install all dependecies in one go.  

The script does the following steps:

1. start by installing [torch](https://pytorch.org/) including torchaudio (preferably with cuda; version used is 2.1.1 with cuda118). 
`pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`

2. `pip install -r requirements.txt`

3. `pip install git+https://github.com/pytorch/fairseq@da8fb630880d529ab47e53381c30ddc8ad235216`

4. install [textlesslib](https://github.com/facebookresearch/textlesslib)

Note: you might get an error beacuse of omegaconf version. this is ok

## Preaparing Datasets

There are 5 files that help create the syntetic files used in the paper:

1. [emovdb_prepare_concat_dataset.py](emovdb_prepare_concat_dataset.py): file used to convert EmoV-DB dataset to the dataset used for the emotion expirement.
2. [emovdb_prepare_concat_gender_dataset.py](emovdb_prepare_concat_gender_dataset.py): file used to convert EmoV-DB dataset to the dataset used for the gender expirement.
3. [iemocap_prepare_concat_dataset.py](iemocap_prepare_concat_dataset.py): file used to convert IEMOCAP dataset to the dataset used for the emotion expirement.
4. [iemocap_prepare_concat_gender_dataset.py](emovdb_prepare_concat_gender_dataset.py): file used to convert IEMOCAP dataset to the dataset used for the gender expirement.
5. [emovdb_prepare_concat_ib_dataset.py](emovdb_prepare_concat_ib_dataset.py): file used to to convert EmoV-DB dataset to test the inductive bias hypothesis.

### runnning the preparation scripts
most files have similar arguments used to run them

- ```-i/--input_folder```: input folder for the datasets
- ```-o/--output_folder```: output folder for the result
- ```-s/--seed```: seed to use for randomness, None is default
- ```-max/--max_concats```: max number of segments for file,default 10
- ```-min/--min_concats```: min number of segments for file,default 2
- ```--num_files```: number of files per speaker/pair
- ```--sample_rate```: resample rate,default 16000
- ```--remove_silence```: will use vad to remove silence in files

#### EmoV-DB unique arguments
- ```--remove_emotions```: emotions to not use for dataset,can be multiple, default None

running EmoV-DB files:

```python emovdb_prepare_concat_dataset.py -i <path to EmoV-DB> -o <output path> -s 42 -min 4 -max 30 --num_files 500 --remove_emotions sleepiness```

```python emovdb_prepare_concat_gender_dataset.py -i <path to EmoV-DB> -o <output path> -s 42 -min 4 -max 30 --num_files 250 --remove_emotions sleepiness```

```python emovdb_prepare_concat_ib_dataset.py -i <path to EmoV-DB> -o <output path> -s 42 -min 4 -max 30 --num_files 500 --remove_emotions sleepiness```

running IEMOCAP files:

```python iemocap_prepare_concat_dataset.py -i <path to IEMOCAP> -o <output path> -s 42 -min 4 -max 30 --num_files 250```

```python iemocap_prepare_concat_gender_dataset.py -i <path to IEMOCAP> -o <output path> -s 42 -min 4 -max 30 --num_files 250```

Note: Those are the commands used to create the datasets on a <strong>linux</strong> computer. results may vary using different operating systems.

## Segmentation/Inference

For inference you will use the file [audio_segment.py](audio_segment.py)

- ```-c/--seg_config```: path to a config file for the segmentor. examples can be seen in [speech_lm/configs](speech_lm/configs). you can also create/update your own config, please read the instructions in [configs README](speech_lm/configs/README.md) 

- ```-j/--input_json```: path to the input json that will be used for inference. each script shown before creates a data.json that can be used for inference. the format for the input is:

```

{
    "key1": {
        "subkey1": {
            "wav_path": <path>
        }
        "subkey2": {
            "wav_path": <path>
        }
    }
    "key2": {
        "subkey1": {
            "wav_path": <path>
        }
        "subkey2": {
            "wav_path": <path>
        }
    }
...
}
```

- `-o/--output_folder`: path where to save the output of the scripts
- `-b/--base_path`: path used for models (or to download the models there)
- `-s/--save_params`: flag if you want to save seg_config.json and data.json used for inference. 
- `--save_audio`: flag if you want to save the segmentation audio files created.

this script will create a <strong>result.json</strong> in the `<output_folder>` folder given.

format will be:

```
{
    "key1": {
        "subkey1": {
            "segmentation": [
                {
                    "start": <start>,
                    "end": <end>
                },
                {
                    "start": <start>,
                    "end": <end>
                },
                ...
            ]
        }
        "subkey2": {
            "segmentation": [
                {
                    "start": <start>,
                    "end": <end>
                },
                {
                    "start": <start>,
                    "end": <end>
                },
                ...
            ]
        }
    }
    ...
}
```
if using `--save_audio` is used, each segment will also have a "wav_path" pointing to the segment wav created.

## Evaluation

To evaluate the inference you can use the file [segment_eval.py](segment_eval.py).

- `-re/--reference_path`: a path to a json file representing the reference (ground truth). format:
```
{
    "key1": {
        "subkey1": {
            "segmentation": [
                {
                    "start": <start>,
                    "end": <end>
                },
                {
                    "start": <start>,
                    "end": <end>
                },
                ...
            ]
        }
        "subkey2": {
            "segmentation": [
                {
                    "start": <start>,
                    "end": <end>
                },
                {
                    "start": <start>,
                    "end": <end>
                },
                ...
            ]
        }
    }
    ...
}
```
- `-hy/--hypothesis_path`: a path to a json file representing the hypothesis (inference). Use same format as reference_path.

- `-m/metrics`: metrics to use. options are: `["coverage","purity", "cpfmeasure", "recall", "precision","rpfmeasure", "r_value"]`. can give multiple options. can also use `"all"` to use all metrics.

- `p/print_sub`: print result for each sample.

- `-o/--output`: optional outputpath used to save results.

- `-ci/--confidence_interval`: action to get confidence intervals as well.

this script will either print a json with the results for the metrics given(unless you use `--output` which will make the script save the json into the given file path)