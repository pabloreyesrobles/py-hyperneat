from hyperneat.substrate import Substrate
from hyperneat.evolution import Hyperneat
from neat.genome import Genome
from neat.neural_network import NeuralNetwork

import os
import sys

ea = Hyperneat()
genome = Genome()
net = NeuralNetwork([], [], 0, 0)

substrate_encoding = open(os.path.join('py-hyperneat', 'tests/config_files/xor_hyperneat_substrate.json'), 'r')
genome_encoding = open(os.path.join('py-hyperneat', 'tests/config_files/testGenome.json'), 'r')

genome.import_genome(genome_encoding)
ea.import_config(substrate_encoding)

ea.build_substrate(genome, net)
net.input([0.0, 1.0])
net.concurrent_activation()
print(net.output())