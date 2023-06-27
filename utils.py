from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import librosa
import numpy as np
import math
import os
import torchaudio
import torch

# Generate dataframes complete (used to plot the graphs for analysis) and min (used for the speaker analysis)
def generateDataFrame(dataset_path, output_path):
    df_complete = pd.DataFrame(columns=['File Path', 'Speaker', 'Gender', 'Audio', 'Pitch', "Pitch Avg",'Raw Duration', 'Speech Duration','Bpm'])
    df_min = pd.DataFrame(columns=['Name', 'Gender', 'Vocal Range', '#F0', 'Lowest Note','Highest Note', 'Total Recorded Hours', 'Speech Recorded Hours'])

    # Load silero-vad model
    model, utils2 = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Creates a tmp dataframe for a specific speaker, calculates all of it's attributes and concatenates it with the complete dataframe
    # Also does the same for the min dataframe
    for speaker in tqdm(os.listdir(dataset_path)):
        df_tmp = pd.DataFrame(columns=['File Path', 'Speaker', 'Gender', 'Audio', 'Pitch', "Pitch Avg",'Raw Duration', 'Speech Duration', 'Bpm'])

        audio_paths = [os.path.join(dataset_path, speaker, fn) for fn in os.listdir(os.path.join(dataset_path, speaker))]
        for audio_path in tqdm(audio_paths):
            y, sr = librosa.load(audio_path)
            raw_duration = librosa.get_duration(y=y, sr=sr)
            speech_duration = calculate_voiced_duration(audio_path, DEVICE, model, utils2)
            # librosa.yin can also be used for a faster analysis but with more inaccurate results
            pitch, _, _ = librosa.pyin(y=y, fmin=librosa.note_to_hz('E1'), fmax=librosa.note_to_hz('C7'))
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=60, units='time')
            df_tmp.loc[len(df_tmp)] = {
                'File Path': audio_path,
                'Speaker': speaker,
                'Gender': '-',
                'Audio': y,
                'Raw Duration': raw_duration,
                'Speech Duration': speech_duration,
                'Bpm': bpm,
                'Pitch': pitch
            }
        
        df_tmp["Pitch"] = df_tmp["Pitch"].apply(lambda x: [value for value in x if not math.isnan(value)])
        df_tmp["Pitch Avg"] = df_tmp["Pitch"].apply(lambda x: np.mean(x))
        df_tmp["Min Pitch"] = df_tmp["Pitch"].apply(lambda x: get_vocal_range(x)[0])
        df_tmp["Max Pitch"] = df_tmp["Pitch"].apply(lambda x: get_vocal_range(x)[1])
        # df_tmp["Speech duration"] = df_tmp["File Path"].apply(calculate_voiced_duration)

        # Formats the dataframe to have only 2 decimal cases and show the actual note
        df_min_tmp = pd.DataFrame(columns=['Name', 'Gender', 'Vocal Range', '#F0', 'Lowest Note', 'Highest Note', 'Total Recorded Hours', 'Speech Recorded Hours'])

        mean_f0 = df_tmp['Pitch Avg'].mean()
        min_pitch = df_tmp['Min Pitch'].min()
        max_pitch = df_tmp['Max Pitch'].max()
        total_raw_duration = df_tmp['Raw Duration'].sum()
        total_speech_duration = df_tmp['Speech Duration'].sum()

        # vocal_range = voical_range_classifier(min_pitch, max_pitch)

        df_min_tmp.loc[len(df_min_tmp)] = {
            'Name': speaker,
            'Gender': '-', 
            'Vocal Range': "-", 
            '#F0': "{:.2f}".format(mean_f0), 
            'Lowest Note': f"{librosa.hz_to_note(min_pitch)}, {'{:.2f}'.format(min_pitch)}",
            'Highest Note': f"{librosa.hz_to_note(max_pitch)}, {'{:.2f}'.format(max_pitch)}",
            'Total Recorded Hours': "{:.2f}".format(total_raw_duration / 3600),
            'Speech Recorded Hours': "{:.2f}".format(total_speech_duration / 3600)
            
        }
        df_complete = pd.concat([df_complete, df_tmp], ignore_index=True)
        df_min = pd.concat([df_min, df_min_tmp], ignore_index=True)

    df_complete.to_pickle(os.path.join(output_path, 'complete_analysis.pkl')) # Saves the types of the columns for future analysis
    df_min.to_csv(os.path.join(output_path, 'summed_analysis.csv'))
    return df_complete, df_min

# Get Voice duration
def calculate_voiced_duration(audio_path, DEVICE, model, utils2):
    model.to(DEVICE)
    (get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks) = utils2

    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.to(DEVICE)
    speech_timestamps = get_speech_timestamps(waveform, model, sampling_rate=16000)
    voiced = sum([t["end"]-t["start"] for t in speech_timestamps])
    return voiced/sample_rate

# Get the vocal range for a specifc audio file (min and max pitch)
def get_vocal_range(pitch_list):
  max_pitch = pitch_list[0]
  min_pitch = pitch_list[0]
  for pitch_value in pitch_list:
    if float(pitch_value) < min_pitch:
      min_pitch = pitch_value
    if float(pitch_value) > max_pitch:
      max_pitch = pitch_value
  return min_pitch, max_pitch

# Classifies the vocal range of the speaker (STILL IN PROGRESS)
def voical_range_classifier(min_hz, max_hz):

  min_note = int(librosa.hz_to_midi(min_hz))
  max_note = int(librosa.hz_to_midi(max_hz))

  soprano = 0
  mezzo_soprano = 0
  alto = 0
  tenor = 0
  baritone = 0
  bass = 0

  for note in range(min_note,max_note):

    if note >  81:
      soprano += 1 #C4 to A5 (60,81)

    if 60 <= note <=  81:
      soprano += 1 #C4 to A5 (60,81)

    if 57 <= note <=  78:
      mezzo_soprano += 1    #A3 to F#5 #(57,78)

    if 55 <= note <=  76:
      alto += 1    #G3 to E5 (and contralto as F3-D5) #(55,76)

    if 48 <= note <=  69:
      tenor += 1    #roughly C3 to A4 #(48,69)

    if 45 <= note <=  65:
      baritone += 1    #A2 to F4 #(45,65)

    if 41 <= note <=  64:
      bass += 1    #F2 to E4 #(41,64)

    if note < 41:
      bass += 1    #F2 to E4 #(41,64)

    dict_classifier = {'soprano': soprano,'mezzo-soprano' : mezzo_soprano,'alto' : alto,'tenor' : tenor,'baritone' : baritone,'bass' : bass}

    inv = {v: k for k, v in dict_classifier.items()}
    classification = inv[max(inv)]

    return classification

# Generates the plots for the analysis
def generatePlots(dataframe, output_path):
    for speaker in tqdm(dataframe["Speaker"].unique()):
        df_tmp = dataframe.loc[dataframe["Speaker"] == speaker]
        sns.set(rc={'figure.figsize':(16,6)})
        ax = sns.scatterplot(data=df_tmp, x="Pitch Avg", y="Bpm", hue="Speaker", s=7, legend=False)
        ax.set_title(speaker)
        ax.set(xlim=(df_tmp["Pitch Avg"].min() - 5, df_tmp["Pitch Avg"].quantile(0.99)))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        fig = ax.get_figure()
        fig.savefig(os.path.join(output_path, speaker + '.png'))

    sns.set(rc={'figure.figsize':(16,6)})
    ax = sns.scatterplot(data=dataframe, x="Pitch Avg", y="Bpm", hue="Speaker", s=7)
    ax.set_title("All Speakers")
    ax.set(xlim=(dataframe["Pitch Avg"].min() - 5, dataframe["Pitch Avg"].quantile(0.99)))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_path, 'All.png'))