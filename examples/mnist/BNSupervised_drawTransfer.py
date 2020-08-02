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

import tkinter as tk
from PIL import Image, ImageDraw, ImageTk 
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt
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

white = (255, 255, 255)

ser = serial.Serial('/dev/ttyUSB0', 115200, stopbits=1, timeout=0.5)

def newplot(data):
    objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
    y_pos = np.arange(len(objects))
    performance = [10,8,6,4,2,1]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Usage')
    plt.title('Programming language usage')

    plt.show()


class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.canvas = tk.Canvas(self, width=450, height=450, bg = "black", cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.button_print = tk.Button(self, text = "Send", command = self.print_points)
        self.button_print.pack(side="top", fill="both", expand=True)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)
        self.image1 = Image.new("RGB", (450, 450), (0,0,0))
        self.draw = ImageDraw.Draw(self.image1)

        self.f = Figure(figsize=(10,10), dpi=45)
        self.a = self.f.add_subplot(111)

        objects = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        y_pos = np.arange(len(objects))
        performance = [0,0,0,0,0,0,0,0,0,0]
        self.a.bar(y_pos, performance, align='center', alpha=0.5, tick_label=objects)

        self.canvas2 = FigureCanvasTkAgg(self.f, self)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

      

    def clear_all(self):
        self.canvas.delete("all")
        self.image1 = Image.new("RGB", (450, 450), (0,0,0))
        self.draw = ImageDraw.Draw(self.image1)



    def print_points(self):
        filename = "netin.png"
        self.image1.save(filename)
        apng = Image.open("netin.png")

        atransIn = transformer(apng)

        periods = ratePeriod(atransIn, time=time, dt=dt)

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

        objects = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        y_pos = np.arange(len(objects))
        performance = output
        self.a.clear()
        self.a.bar(y_pos, performance, align='center', alpha=0.5, tick_label=objects)
        self.canvas2.draw()
        

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    def draw_from_where_you_are(self, event):
        if self.points_recorded:
            self.points_recorded.pop()
            self.points_recorded.pop()

        self.x = event.x
        self.y = event.y
        self.canvas.create_line(self.previous_x, self.previous_y, 
                                self.x, self.y,fill="white", width = 40)
        self.draw.line([self.previous_x, self.previous_y, 
                                self.x, self.y], white, width=40)
        self.points_recorded.append(self.previous_x)
        self.points_recorded.append(self.previous_y)
        self.points_recorded.append(self.x)     
        self.points_recorded.append(self.x)        
        self.previous_x = self.x
        self.previous_y = self.y

if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()

