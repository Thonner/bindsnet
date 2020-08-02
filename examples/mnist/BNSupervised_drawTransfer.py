import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import serial
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Sequence, Iterable

from bindsnet.datasets import create_torchvision_dataset_wrapper
from bindsnet.encoding.encodings import ratePeriod

from bindsnet.datasets import MNIST
from bindsnet.encoding import RatePeriod
from bindsnet.analysis.plotting import (
    plot_input,
    plot_assignments,
    plot_performance,
    plot_weights,
    plot_spikes,
    plot_voltages,
)

from PIL import Image

import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--n_neurons", type=int, default=200)
parser.add_argument("--time", type=int, default=500)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)

args = parser.parse_args()


seed = args.seed
n_neurons = args.n_neurons
time = args.time
dt = args.dt
intensity = args.intensity

torch.manual_seed(seed)

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
per_class = int(n_neurons / 10)

transformer = transforms.Compose(
        [transforms.Grayscale(),transforms.Resize(size=(22,22)), transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    )

RateEncoder = RatePeriod(time=time, dt=dt)

apng = Image.open("test0.png")

atransIn = transformer(apng)

periods = ratePeriod(atransIn, time=time, dt=dt)


ser = serial.Serial('/dev/ttyUSB0', 115200, stopbits=1, timeout=0.5)
sCnt = 0
output = [0]*10
outputall = [0]*200

start = ser.read()

for j in range(22*22):

    abyte = bytes([(j >> 8)])
    ser.write(abyte)
    abyte = bytes([j & 0xff])
    ser.write(abyte)
    abyte = bytes([(periods[j].int().item() >> 8) & 0xff])
    ser.write(abyte)
    abyte = bytes([periods[j].int().item() & 0xff])
    ser.write(abyte)

while(sCnt != 2):

    newbyte = ser.read()
    if (newbyte == b''):
        sCnt += 1
    else:
        newint = int(newbyte.hex(), 16)
        if (newint == 255):
            sCnt += 1
        else:
            output[newint//(200//10)] += 1
            outputall[newint] += 1


maxInd = output.index(max(output))

print(output)

print("hej")