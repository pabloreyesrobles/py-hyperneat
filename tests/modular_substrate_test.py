from hyperneat.spatial_node import SpatialNode, SpatialNodeType
from hyperneat.substrate import Substrate
from hyperneat.evolution import Hyperneat

from neat.genes import ConnectionGene, NodeGene, NodeType
from neat.genome import Genome
from neat.activation_functions import ActivationFunction
from neat.neural_network import NeuralNetwork

import os
import sys

# Genome
genome = Genome(num_layers=15, weights_range=[-3.0, 3.0])
genome.create_genome_by_size(8, 3)
net = genome.build_phenotype()
net.print_connections()
sys.exit()

# Substrate setting
# Init substrate set
substrate_set = []
for i in range(2):
	s = Substrate()

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

ea = Hyperneat()
net = NeuralNetwork([], [], 0, 0)

substrate_encoding = open(os.path.join('py-hyperneat', 'tests/config_files/xor_hyperneat_substrate.json'), 'r')
genome_encoding = open(os.path.join('py-hyperneat', 'tests/config_files/testGenome.json'), 'r')

genome.import_genome(genome_encoding)
ea.import_config(substrate_encoding)

ea.build_substrate(genome, net)
net.input([0.0, 1.0])
net.concurrent_activation()
print(net.output())