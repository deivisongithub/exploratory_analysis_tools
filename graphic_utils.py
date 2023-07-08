import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import librosa

def hist_mean_F0(df,output_path):

  ax1 = sns.histplot(data = df,x=df['Pitch Avg'],hue=df['Gender'],multiple="stack")

  ax1.set_xlabel('F0')
  ax1.set_ylabel('Frequency')
  ax1.set_title('Stacked histogram of mean F0')
  fig = ax1.get_figure()
  fig.savefig(output_path + '/' + 'Stacked_histogram_of_mean_F0.png')
  ax1.clear()

  return "---graphic done---"

def hist_bpm(df,output_path):
  
  ax2 = sns.histplot(data = df,y=df['Bpm'],hue=df['Gender'],multiple="stack")

  ax2.set_xlabel('Frequency')
  ax2.set_ylabel('Bpm')
  ax2.set_title('Stacked histogram of Bpm')
  fig = ax2.get_figure()
  fig.savefig(output_path + '/' + 'hist_bpm.png')
  ax2.clear()

  return "---graphic done---"

def hist_distribution_F0(df,output_path):

  ax3 = sns.scatterplot(data=df, x=df['Pitch Avg'], y=df['Bpm'], hue=df['Gender'])

  ax3.set_xlabel('F0')
  ax3.set_ylabel('Bpm')
  ax3.set_title('Distribution of F0 mean in relation to Bpm')
  fig = ax3.get_figure()
  fig.savefig(output_path + '/' + 'hist_distribution_F0.png')
  ax3.clear()

  return "---graphic done---"

def distribution_of_pitch_gender(df,output_path):

  Midi_notes_M = []
  xM = df[df['Gender'] == 'M']
  xM = xM['Pitch Filtered']
  dict_M = {}
  index = 0
  for i in xM:
    for j in i:
      dict_M[f"row{index}"] = int(librosa.hz_to_midi(j)),'M'
      index += 1

  dfm = pd.DataFrame.from_dict(dict_M, orient='index',columns=['note','Gender'])
  
  Midi_notes_F = []
  xF = df[df['Gender'] == 'F']
  xF = xF['Pitch Filtered']
  dict_F = {}
  index = 0
  for i in xF:
    for j in i:
      dict_F[f"row{index}"] = int(librosa.hz_to_midi(j)),'F'
      index += 1
  
  dff = pd.DataFrame.from_dict(dict_F, orient='index',columns=['note','Gender'])

  dfconcat = pd.concat([dfm, dff])

  #plt.figure(figsize=(20,5))
  sns.set(rc={'figure.figsize':(20,5)})
  ax4 = sns.histplot(data=dfconcat,x='note',hue='Gender',multiple="dodge",bins=max(dfconcat['note']))

  ax4.set_xlabel('Pitch Note')
  ax4.set_ylabel('Frequency')
  ax4.set_title('Stacked histogram of Midi note')
  x_ticks = [i for i in range(min(dfconcat['note']),max(dfconcat['note'])+1)]
  ax4.set_xticks(x_ticks)
  fig = ax4.get_figure()
  fig.savefig(output_path + '/' + 'Stacked_histogram_of_midi_note.png')
  ax4.clear()

  return "---graphic done---"

def hist_vr(df,output_path):
  
  sns.set(rc={'figure.figsize':(16,6)})
  ax = sns.histplot(data = df,x=df['Vocal Range'],multiple="stack")
  ax.set_xlabel('Vocal Range')
  ax.set_ylabel('Frequency')
  ax.set_title('histogram of Vocal Range')
  x_ticks = ['Bass','Baritone','Tenor','Alto','Mezzo-soprano','Soprano']
  ax.set_xticks(x_ticks)
  fig = ax.get_figure()
  fig.savefig(output_path + '/' + 'hist_vr.png')
  ax.clear()

  return "---graphic done---"