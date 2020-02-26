from population import Population
from neural_network import NeuralNetwork
from genome import Genome
import numpy as np
import math
import sys

params = open('testConfig.json', 'r')
genome = open('testXor.json', 'r')

pop = Population()
pop.start_generation(genome, params)

net = pop.organisms[0].build_layered_phenotype()
net.input([1, 0])
net.activate()
print(net.output())