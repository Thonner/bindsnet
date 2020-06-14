import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.nn import Module, Parameter
from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Sequence, Iterable

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.encoding import RateEncoder
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

'''
Network describtion used for the final Showcase network. It is a slightly modified 
and scalled version of the Diehl and Cook network found in 
bindsnet/network/network.py.
numeric scaling of the network increase chance of succes when porting to integer implemenation 
with rounding of floating point parameters.
'''
class ShowCaseNet(Network): 
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

def toLIF(network : Network): # was not used for final implementation
    new_network = Network(dt=1, learning=True)
    input_layer = Input(
        n=network.X.n, shape=network.X.shape, traces=True, tc_trace=network.X.tc_trace.item()
    )
    exc_layer = LIFNodes(
        n=network.Ae.n,
        traces=True,
        rest=network.Ai.rest.item(),
        reset=network.Ai.reset.item(),
        thresh=network.Ai.thresh.item(),
        refrac=network.Ai.refrac.item(),
        tc_decay=network.Ai.tc_decay.item(),
    )
    inh_layer = LIFNodes(
        n=network.Ai.n,
        traces=False,
        rest=network.Ai.rest.item(),
        reset=network.Ai.reset.item(),
        thresh=network.Ai.thresh.item(),
        tc_decay=network.Ai.tc_decay.item(),
        refrac=network.Ai.refrac.item(),
    )

    # Connections
    w = network.X_to_Ae.w
    input_exc_conn = Connection(
        source=input_layer,
        target=exc_layer,
        w=w,
        update_rule=PostPre,
        nu=network.X_to_Ae.nu,
        reduction=network.X_to_Ae.reduction,
        wmin=network.X_to_Ae.wmin,
        wmax=network.X_to_Ae.wmax,
        norm=network.X_to_Ae.norm * 1,
    )
    w = network.Ae_to_Ai.w
    exc_inh_conn = Connection(
        source=exc_layer, target=inh_layer, w=w, wmin=network.Ae_to_Ai.wmin, wmax=network.Ae_to_Ai.wmax
    )
    w = network.Ai_to_Ae.w
    
    inh_exc_conn = Connection(
        source=inh_layer, target=exc_layer, w=w, wmin=network.Ai_to_Ae.wmin, wmax=network.Ai_to_Ae.wmax
    )

    # Add to network
    new_network.add_layer(input_layer, name="X")
    new_network.add_layer(exc_layer, name="Ae")
    new_network.add_layer(inh_layer, name="Ai")
    new_network.add_connection(input_exc_conn, source="X", target="Ae")
    new_network.add_connection(exc_inh_conn, source="Ae", target="Ai")
    new_network.add_connection(inh_exc_conn, source="Ai", target="Ae")

    exc_voltage_monitor = Monitor(new_network.layers["Ae"], ["v"], time=500)
    inh_voltage_monitor = Monitor(new_network.layers["Ai"], ["v"], time=500)
    new_network.add_monitor(exc_voltage_monitor, name="exc_voltage")
    new_network.add_monitor(inh_voltage_monitor, name="inh_voltage")

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(new_network.layers[layer], state_vars=["s"], time=time)
        new_network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    return new_network


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--n_neurons", type=int, default=200)
parser.add_argument("--n_train", type=int, default=2000)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_clamp", type=int, default=1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=22.5)
parser.add_argument("--time", type=int, default=500)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=100)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--run_only", dest="run_only", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True, , run_only = True)

args = parser.parse_args()


seed = args.seed
n_neurons = args.n_neurons
n_train = args.n_train
n_test = args.n_test
n_clamp = args.n_clamp
exc = args.exc
inh = args.inh
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
run_only = args.run_only

if gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
per_class = int(n_neurons / 10)



# Build Diehl & Cook 2015 network.
network = ShowCaseNet(
    n_inpt=22*22,
    n_neurons=n_neurons,
    exc=exc*1000,
    inh=inh*1000,
    dt=dt,
    norm=78.4*1000,
    nu=[0, 1e-2],
    inpt_shape=(1, 22, 22),
)


# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Load MNIST data.
dataset = MNIST(
    RateEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(size=(22,22)), transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons)

# Neuron assignments and spike proportions.
assignments = -torch.ones_like(torch.Tensor(n_neurons))
proportions = torch.zeros_like(torch.Tensor(n_neurons, 10))
rates = torch.zeros_like(torch.Tensor(n_neurons, 10))


# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Labels to determine neuron assignments and spike proportions and estimate accuracy
labels = torch.empty(update_interval)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

if gpu:
    network.to("cuda")

if run_only:
    # Modify the network to get as close to hardware implementation as easily possible
    network_old = pickle.load(open("examples/mnist/pickledNetworks/SupervisedNuScaled.p", "rb"))
    network_old.learning = False
    network_old.Ae.learning = False
    network_old.Ae_to_Ai.training = False
    network_old.Ai.learning = False
    network_old.Ai_to_Ae.training = False
    network_old.X.learning = False
    network_old.X_to_Ae.training = False
    network_old.X_to_Ae.w = Parameter(network_old.X_to_Ae.w.round(), requires_grad=False)
    network_old.X_to_Ae.norm = None
    network_old.Ae.one_spike = False
    network_old.Ai.decay = torch.tensor(0.875)
    network_old.Ae.decay = torch.tensor(.9921875) 
    network_old.Ae.theta = network_old.Ae.theta.round() 


    hit = 0
    confusion = []
    for i in range(10):
        confusion.append([0,0,0,0,0,0,0,0,0,0])
    spikesum = 0.0
    pbar = tqdm(enumerate(dataloader))
    for (i, datum) in pbar:
        if i > 500:
            break
        image = datum["encoded_image"]
        label = datum["label"]
        inputs = {"X": image.view(time, 1, 1, 22, 22)}
        network_old.run(inputs=inputs, time=time)


        out_spikes = network_old.monitors["Ae_spikes"].get("s")
        class_spike=torch.zeros(10)
        for k in range(time):
            for j in range(n_neurons):
                #print(out_spikes[i,0,j])
                if out_spikes[k,0,j]:
                    class_spike[j//(n_neurons//10)] += 1
        
        aVal, maxInd = class_spike.max(0)
        if maxInd == label[0]:
            hit += 1
        confusion[label[0].item()][maxInd.item()] += 1
        print(label[0])
        print(class_spike)
        spikesum += class_spike.sum().item()
        # print(out_spikes)

    acc = hit/500
    print("\n accuacy: " + str(acc) +"\n")
    print("confusion:")
    for i in range(10):
        print(confusion[i])
    print("avg nr of spikes:" + str(spikesum/500))
else: 
    # Train the network.
    print("Begin training.\n")

    inpt_axes = None
    inpt_ims = None
    spike_axes = None
    spike_ims = None
    weights_im = None
    assigns_im = None
    perf_ax = None
    voltage_axes = None
    voltage_ims = None

    pbar = tqdm(enumerate(dataloader))
    for (i, datum) in pbar:
        if i > n_train:
            break

        image = datum["encoded_image"]
        label = datum["label"]
        pbar.set_description_str("Train progress: (%d / %d)" % (i, n_train))

        if i % update_interval == 0 and i > 0:
            input_exc_weights = network.connections[("X", "Ae")].w
            w_arg = 0.0
            for i in range(22*22):
                for j in range(n_neurons):
                    w_arg += input_exc_weights[i, j]
            print(w_arg/((22*22)*n_neurons))
            # Get network predictions.
            all_activity_pred = all_activity(spike_record, assignments, 10)
            proportion_pred = proportion_weighting(
                spike_record, assignments, proportions, 10
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100 * torch.sum(labels.long() == all_activity_pred).item() / update_interval
            )
            accuracy["proportion"].append(
                100 * torch.sum(labels.long() == proportion_pred).item() / update_interval
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]))
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(spike_record, labels, 10, rates)

        #Add the current label to the list of labels for this update_interval
        labels[i % update_interval] = label[0]

        # Run the network on the input.
        choice = np.random.choice(int(n_neurons / 10), size=n_clamp, replace=False)
        clamp = {"Ae": per_class * label.long() + torch.Tensor(choice).long()}
        inputs = {"X": image.view(time, 1, 1, 22, 22)}
        network.run(inputs=inputs, time=time, clamp=clamp)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[i % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)
        '''class_spike=torch.zeros(10)
        print(label[0])
        for i in range(time):
            for j in range(n_neurons):
                if spikes["Ae"].get("s")[i,0,j]:
                    class_spike[j//10] += 1
        
        print("\nlabel\n")
        print(class_spike)
        '''

        # Optionally plot various simulation information.
        if plot:
            inpt = inputs["X"].view(time, 22*22).sum(0).view(22, 22)
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view((22*22), n_neurons), n_sqrt, 22
            )
            
            square_assignments = get_square_assignments(assignments, n_sqrt)
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}

            inpt_axes, inpt_ims = plot_input(
                image.sum(1).view(22, 22), inpt, label=label, axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(
                {layer: spikes[layer].get("s").view(time, 1, -1) for layer in spikes},
                ims=spike_ims,
                axes=spike_axes,
            )
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes
            )

            plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.

    pickle.dump(network, open( "ANewNetwork.p", "wb" ) )

    print("Progress: %d / %d \n" % (n_train, n_train))
    print("Training complete.\n")
