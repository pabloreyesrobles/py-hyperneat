from neat.population import Population
from neat.neural_network import NeuralNetwork
from neat.genome import Genome
from neat.neat import Neat, TrainTask

import math
import sys
import numpy as np
import matplotlib.pyplot as plt

def fitness(input_data, net):
    error = 0.0
    outputs = []

    logic = input_data[0]
    net.reset_values()
    net.input(logic)
    net.activate_net()
    error += math.fabs(net.output()[0])
    outputs.append(net.output()[0])

    logic = input_data[1]
    net.reset_values()
    net.input(logic)
    net.activate_net()
    error += math.fabs(1 - net.output()[0])
    outputs.append(net.output()[0])

    logic = input_data[2]
    net.reset_values()
    net.input(logic)
    net.activate_net()
    error += math.fabs(1 - net.output()[0])
    outputs.append(net.output()[0])

    logic = input_data[3]
    net.input(logic)
    net.activate_net()
    error += math.fabs(net.output()[0])
    outputs.append(net.output()[0])

    #print('{:f}, {:f}, {:f}, {:f}'.format(outputs[0], outputs[1], outputs[2], outputs[3]))
    return math.pow(4 - error, 2)

params = open('config_files/testConfig.json', 'r')
genome = open('config_files/testXor.json', 'r')

evolution = Neat(fitness_eval=fitness, train_task=TrainTask.PREDICTION)
evolution.import_config(params, genome)
#evolution.pop.activation_set.set_atemporal_set()
#evolution.pop.activation_set.unset_lin_group()
#evolution.pop.activation_set.use_only_sigmoid()
evolution.pop.activation_set.use_only_tanh()

xor_eval = [[0, 0], [0, 1], [1, 0], [1, 1]]

for i in range(evolution.max_generation):
    evolution.set_multi_input(xor_eval)
    evolution.evaluate_population()    
    evolution.epoch()

evolution.pop.champion_genome.save_genome()
net = evolution.pop.champion_genome.build_layered_phenotype()

generations = np.linspace(0, 99, 100)
