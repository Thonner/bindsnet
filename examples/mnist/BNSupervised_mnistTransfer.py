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

# Load MNIST data.
dataset = MNIST(
    RatePeriod(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(size=(22,22)), transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0
)



hit = 0
confusion = []
for i in range(10):
    confusion.append([0,0,0,0,0,0,0,0,0,0])
ser = serial.Serial('/dev/ttyUSB1', 115200, stopbits=1, timeout=0.5)
pbar = tqdm(enumerate(dataloader))
for (i, datum) in pbar:
    if i > 1000:
        break

    image = datum["encoded_image"]
    label = datum["label"]

    sCnt = 0
    output = [0]*10
    outputall = [0]*200

    start = ser.read()

    for j in range(22*22):

        abyte = bytes([(j >> 8)])
        ser.write(abyte)
        abyte = bytes([j & 0xff])
        ser.write(abyte)
        abyte = bytes([(image[0,j].int().item() >> 8) & 0xff])
        ser.write(abyte)
        abyte = bytes([image[0,j].int().item() & 0xff])
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
    if maxInd == label[0].item():
        hit += 1
    confusion[label[0].item()][maxInd] += 1
    print("label: "+str(label.item())+"\n")
    print(output)

ser.close()
acc = hit/1000
print("\n accuacy: " + str(acc) +"\n")
print("confusion:")
for i in range(10):
    print(confusion[i])
    

