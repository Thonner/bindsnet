import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Sequence, Iterable

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_assignments,
    plot_performance,
    plot_weights,
    plot_spikes,
    plot_voltages,
)
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes
from bindsnet.network.topology import Connection, LocalConnection

import pickle
import json

class ShowCaseNet(Network): # Definition of the slightly modified Diehl & Cook found in bindsnet/network/network.py
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1000.0,
        norm: float = 78.4,
        theta_plus: float = 0.05*1000,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=0.0,
            reset=5.0,
            thresh=12.0*1000,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=0.0,
            reset=15.0,
            thresh=20.0*1000,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3*1000 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")


def getDecayShift(modelDecay):
    for i in range(1,8):
        if (1 - modelDecay > 0.5):
            return 1
            break
        
        if (1 - modelDecay < 1/(2**i) and 1 - modelDecay > 1/(2**(i+1))):
            if ((1 - modelDecay) - (1/(2**(i+1))) > (1/(2**i) - (1/(2**(i+1))))/2 ):
                return i
            else:
                return i+1
            break


# The actual mapping starts here. Generates json file to be read during by scala to map network during hardware generation
network_old = pickle.load(open("examples/mnist/pickledNetworks/SupervisedNuScaled.p", "rb"))
network_old.learning = False
network_old.Ae.learning = False
network_old.Ae_to_Ai.training = False
network_old.Ai.learning = False
network_old.Ai_to_Ae.training = False
network_old.X.learning = False
network_old.X_to_Ae.training = False
AeThresh = [network_old.Ae.thresh.item()] * network_old.Ae.n
Aetheta  = network_old.Ae.theta.tolist()
for i in range(network_old.Ae.n):
    AeThresh[i] = AeThresh[i] + Aetheta[i]
AeBiases1 = network_old.X_to_Ae.b.data.tolist()
AeBiases2 = network_old.Ai_to_Ae.b.data.tolist()
for i in range(network_old.Ae.n):
    AeBiases1[i] = AeBiases1[i] + AeBiases2[i]
AeDecay =  getDecayShift(network_old.Ae.decay.item())

networkData = {}
networkData['l1'] = []
networkData['l1'].append({
    'reset'  : [round(network_old.Ae.reset.item())] * network_old.Ae.n,
    'thresh' : [round(numb) for numb in AeThresh],
    'refrac' : [network_old.Ae.refrac.item()] * network_old.Ae.n,
    'decay'  : [AeDecay] * network_old.Ae.n,
    'biases' : [round(numb) for numb in AeBiases1],
    'w1size' : list(network_old.X_to_Ae.w.data.size()),
    'w1'     : [([round(numb) for numb in numb2]) for numb2 in network_old.X_to_Ae.w.data.tolist()],
    'w2size' : list(network_old.Ai_to_Ae.w.data.size()),
    'w2'     : [([round(numb) for numb in numb2]) for numb2 in network_old.Ai_to_Ae.w.data.tolist()],
})

AiDecay = getDecayShift(network_old.Ai.decay.item())
networkData['l2'] = []
networkData['l2'].append({
    'reset'  : [round(network_old.Ai.reset.item())] * network_old.Ai.n,
    'thresh' : [round(network_old.Ai.thresh.item())] * network_old.Ai.n,
    'decay'  : [AiDecay] * network_old.Ae.n,
    'refrac' : [network_old.Ae.refrac.item()] * network_old.Ae.n,
    'biases' : [round(numb) for numb in  network_old.Ae_to_Ai.b.data.tolist()],
    'wsize'  : list(network_old.Ae_to_Ai.w.data.size()),
    'w'      : [([round(numb) for numb in numb2]) for numb2 in network_old.Ae_to_Ai.w.data.tolist()],
})


with open('examples/mnist/mapping/networkData.json', 'w') as jsonfile:
    json.dump(networkData, jsonfile)



