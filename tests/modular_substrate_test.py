from hyperneat.spatial_node import SpatialNode, SpatialNodeType
from hyperneat.substrate import Substrate
from hyperneat.evolution import Hyperneat

from neat.genes import ConnectionGene, NodeGene, NodeType
from neat.genome import Genome
from neat.activation_functions import ActivationFunction
from neat.neural_network import NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Genome
genome = Genome(num_layers=15, weights_range=[-1.0, 1.0])
genome.create_genome_by_size(8, 3)
net = genome.build_phenotype()

# Substrate setting
# Init substrate set
substrate_set = []
for i in range(2):
	s = Substrate()
	s.activation_function = ActivationFunction().get('TANH')

	# Must create new objects or deep copies
	s.input_nodes = [SpatialNode(0, SpatialNodeType.INPUT, [0.0, -0.5], ActivationFunction().get('TANH'), 0)]
	s.output_nodes = [SpatialNode(1, SpatialNodeType.OUTPUT, [-0.5, 0.5], ActivationFunction().get('TANH'), 2),
					  SpatialNode(2, SpatialNodeType.OUTPUT, [0.5, 0.5], ActivationFunction().get('TANH'), 2)]
	s.hidden_nodes = [SpatialNode(3, SpatialNodeType.HIDDEN, [-0.5, 0.0], ActivationFunction().get('TANH'), 1),
					  SpatialNode(4, SpatialNodeType.HIDDEN, [0.5, 0.0], ActivationFunction().get('TANH'), 1)]

	s.input_count = 1
	s.output_count = 2
	s.hidden_count = 2

	s.extend_nodes_list()
	substrate_set.append(s)

module_conn = [[0, 1], [0, 2], [0, 3], [0, 4], [3, 1], [3, 2], [3, 4], [4, 1], [4, 2], [4, 3]]
inter_substrate_conn = [[0, 4, 1, 3], [1, 3, 0, 4], [0, 2, 2, 1], [1, 1, 2, 2]]

hid_out_sub = Substrate()
hid_out_sub.activation_function = ActivationFunction().get('TANH')

hid_out_sub.output_nodes = [SpatialNode(0, SpatialNodeType.OUTPUT, [0.0, 0.5], ActivationFunction().get('TANH'), 1)]
hid_out_sub.hidden_nodes = [SpatialNode(1, SpatialNodeType.HIDDEN, [-0.5, -0.5], ActivationFunction().get('TANH'), 0),
							SpatialNode(2, SpatialNodeType.HIDDEN, [0.5, -0.5], ActivationFunction().get('TANH'), 0)]
hid_out_sub.extend_nodes_list()

hid_out_conn = [[1, 0], [2, 0], [1, 2], [2, 1]]
substrate_set.append(hid_out_sub)

substrate_set[0].coordinates = (-0.75, -0.75)
substrate_set[1].coordinates = (0.75, -0.75)
substrate_set[2].coordinates = (0.0, 0.75)

intra_conn_set = []
for _ in range(2):
	intra_conn_set.append(module_conn)
intra_conn_set.append(hid_out_conn)

ea = Hyperneat()
ea.connection_threshold = 0.1
ea.max_connection_weight = 2.0
ea.max_bias = 0.5
ea.max_delay = 0.8

net = ea.build_modular_substrate(genome, substrate_set, intra_conn_set, inter_substrate_conn)
net.reset_values()

time = np.linspace(0, 6, int(6 / 0.05), endpoint=False)
freq = np.linspace(0, 1, int(6 / 0.05), endpoint=False)
freq[60:] = 1 - freq[60:]
signal_1 = np.sin(time * 2 * np.pi * freq)
signal_2 = np.cos(time * 2 * np.pi * freq)

output_signal = np.zeros([5, time.shape[0]])
input_signal = np.zeros(time.shape[0])
out_id = net.out_neurons

for t, _ in enumerate(time):
	net.input([signal_1[t], signal_2[t]])
	net.activate_net(0.05)
	input_signal[t] = net.neurons[net.in_neurons[0]].output

	for o, oid in enumerate(out_id):
		output_signal[o, t] = net.neurons[oid].output

net.print_connections()

fig, ax = plt.subplots(3, 3)
ax[0, 0].plot(output_signal[0])
ax[0, 0].plot(output_signal[1])
ax[0, 0].plot(output_signal[2])
ax[0, 0].plot(output_signal[3])

ax[0, 1].plot(output_signal[1])
ax[0, 1].plot(output_signal[2])
ax[0, 1].plot(output_signal[4])

ax[0, 2].plot(input_signal)

ax[1, 0].plot(output_signal[0])
ax[1, 1].plot(output_signal[1])
ax[1, 2].plot(output_signal[2])

ax[2, 0].plot(output_signal[3])
ax[2, 1].plot(output_signal[4])

ax[2, 2].plot(input_signal)
ax[2, 2].plot(output_signal[0])

plt.tight_layout()
plt.show()