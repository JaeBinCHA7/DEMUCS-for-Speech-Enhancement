import os
import soundfile
import numpy as np


def scan_directory(dir_name):
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()

    addrs = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".wav"):
                filepath = subdir + file
                addrs.append(filepath)
    return addrs


def find_pair(noisy_file_name):
    clean_dirs = []
    for i in range(len(noisy_file_name)):
        addrs = noisy_file_name[i]
        if addrs.endswith(".wav"):
            clean_addrs = str(addrs).replace('noisy', 'clean')
            clean_dirs.append(clean_addrs)
    return clean_dirs


def addr2wav(addr):
    wav, fs = soundfile.read(addr)
    # normalize
    wav = minMaxNorm(wav)
    return wav


def minMaxNorm(wav, eps=1e-8):
    max_v = np.max(abs(wav))
    min_v = np.min(abs(wav))
    wav = (wav - min_v) / (max_v - min_v + eps)
    return wav


# make a new dir
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)