import pandas
import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import tkinter as tk

import math
import scipy.stats
import os
import random

path = "./Samples"  # path to sound folders and files
current_noise = 0
PLAYAUDIOS = False
PLOT_THREE_SAMPLES = False
MFCC_COMPONENTS = 3

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '\\' + filename

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="wav")

    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i + min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')


def plot_mfcc_low_middle_high(mfcc):
    y_low, sr_low = librosa.load(str(path + "/Highlights/LOW2.wav"))
    y_middle, sr_middle = librosa.load(str(path + "/PotentialGoalMoves/MIDDLE2.wav"))
    y_high, sr_high = librosa.load(str(path + "/GoalMoves/HIGH2.wav"))
    mfcc_low = librosa.feature.mfcc(y_low, sr_low, n_mfcc=MFCC_COMPONENTS, dct_type=2)
    mfcc_middle = librosa.feature.mfcc(y_middle, sr_middle, n_mfcc=MFCC_COMPONENTS, dct_type=2)
    mfcc_high = librosa.feature.mfcc(y_high, sr_high, n_mfcc=MFCC_COMPONENTS, dct_type=2)

    # Plot MFCCs
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].set(title='MFCCs of Audience Noise in Highlights')
    i = librosa.display.specshow(mfcc_low, x_axis='time', ax=ax[0])
    ax[1].set(title='MFCCs of Audience Noise in Potential Goal Moves')
    librosa.display.specshow(mfcc_middle, x_axis='time', ax=ax[1])
    ax[2].set(title='MFCCs of Audience Noise in GoalMoves')
    librosa.display.specshow(mfcc_high, x_axis='time', ax=ax[2])
    plt.colorbar(i)
    plt.show()


def plot_mfcc(mfcc):
    # Plot MFCCs
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].set(title='MFCCs of Low Audience Noise')
    i = librosa.display.specshow(mfcc, x_axis='time', ax=ax[0])
    ax[1].set(title='MFCCs of Audience Noise in PotentialGoalMoves')
    librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
    ax[2].set(title='MFCCs of Audience Noise in GoalMoves')
    librosa.display.specshow(mfcc, x_axis='time', ax=ax[2])
    plt.colorbar(i)
    plt.show()


def describe_freq(freqs):
    mean = np.mean(freqs)
    std = np.std(freqs)
    maxv = np.amax(freqs)
    # mean = np.amin(freqs)
    median = np.median(freqs)
    skew = scipy.stats.skew(freqs)
    kurt = scipy.stats.kurtosis(freqs)
    q1 = np.quantile(freqs, 0.25)
    q3 = np.quantile(freqs, 0.5)
    # print(mean, std, maxv, mean, median, skew, kurt, q1, q3)
    print(mean, std, maxv, mean, median, q1, q3)

    global current_noise
    if (mean < -50):
        current_noise = 1
        print('1')
    elif mean > -50 and mean < -15:
        current_noise = 2
        print('2')
    elif mean > -15:
        current_noise = 3
        print('3')
'''    
    if (std < 0.05):
        current_noise = 1
        print('1')
    elif maxv > 0.05 and maxv < 0.1:
        current_noise = 2
        print('2')
    elif maxv > 0.1:
        current_noise = 3
        print('3')
'''




def keyboardPress(noise):
    audioSelect = random.choice(os.listdir(path + "/" + noise))  # selects random sound
    keypress = str(path + "/" + noise + "/" + audioSelect)  # gets the path to the random sound
    print(keypress)
    audio_sample = AudioSegment.from_wav(keypress)
    if PLAYAUDIOS:
        assert isinstance(audio_sample, object)
        play(audio_sample)
    y, sr = librosa.load(keypress)
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=MFCC_COMPONENTS, dct_type=2)

    # describe_freq(y)
    describe_freq(mfcc)

    if PLOT_THREE_SAMPLES:
        plot_mfcc_low_middle_high(mfcc)


def key_handler(event=None):
    if event and event.keysym in ('1', 'p'):
        keyboardPress("Highlights")
    if event and event.keysym in ('2', 'p'):
        keyboardPress("PotentialGoalMoves")
    if event and event.keysym in ('3', 'p'):
        keyboardPress("GoalMoves")


if __name__ == '__main__':
    # creating interface for key reading with tkinter
    r = tk.Tk()
    t = tk.Text()

    print("press 1 to random low noise")
    print("press 2 to random PotentialGoalMoves noise")
    print("press 3 to random GoalMoves noise")
    print("press p to random noise")
    t.pack()
    r.bind('<Key>', key_handler)
    r.mainloop()
