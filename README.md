# Dataset Analysis

This python script analyses a dataset a generates a chart as well as some plots with data from it, the current extracted information is (in a speaker basis):
* Name
* Gender
* Vocal Range
* #F0
* Lowest Note
* Highest Note
* Total Recorded Hours
* Speech Recorded Hours

The scripts works on a dataset that has subdirectories for each speaker, each one containing the audio files from that speaker, so if the dataset isn't following this sctruture, it has to be formatted.

```
dataset
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

## Dependencies
This script requires the following libraries:

* librosa==0.9.2
* matplotlib==3.5.1
* numpy==1.21.5
* pandas==1.4.3
* seaborn==0.11.2
* torch==1.13.1+cu116
* torchaudio==0.13.1+cu116
* tqdm==4.64.0

## How to use
To execute the dataset analysis script, just use the following command:
```
python main.py --dataset_path teste --output_path output                      
```

## Notes
This script was written in Python 3.9,