#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import wave
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt  
from matplotlib import cm
from matplotlib import animation
import math
import time
import random
from pyaudio import PyAudio,paInt16

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

from scipy.fftpack import fft
from scipy.fftpack import ifft

from pydub import AudioSegment
import configparser
import re
from pypinyin import pinyin, lazy_pinyin, Style
import shutil
import sox

g_wave_data = []
g_samwid = 0
g_fs = 0
JUMP_STEP = 15
usebaidu = True
usexf = False #xunfei
output_trn_only = False
maxduration = 100
minduration = 3
skipcount = 0
confirmmute_value = 0
jump_period = 1.0
lrc_nouse_head = ""
g_txt_lrc_lines = []
mute_list = []
enlarge_audio = 1.0
out_path = ""
debugpath = ""

def upsample_wav(file1, out_path, rate):
    tfm = sox.Transformer()
    tfm.rate(rate)
    tfm.build(file1, out_path)
    return out_path

def downsampleto_16000_wav(src,dst):
    ext = src[-3:]
    if ext == "mp3":
        tmpwav = open_mp3(src)
        src = tmpwav
    upsample_wav(src, dst, 16000)

def save_wave_file(filename, data, channels, sampwidth, framerate):
    '''save the date to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

if(__name__=='__main__'):
    argc = len(sys.argv)
    if argc != 3:
        print("usuage:", sys.argv[0], " mp3filename outputwav_file_name")
        print("example: python3 .\\transwav.16000.py .\sam.wav .\sam.16000.wav")
        exit(1)
    downsampleto_16000_wav(sys.argv[1], sys.argv[2])    
