from hyperneat.substrate import Substrate
from hyperneat.evolution import Hyperneat
from neat.genome import Genome
from neat.neural_network import NeuralNetwork
from neat.evolution import Neat, TrainTask
from neat.activation_functions import ActivationFunction

import os
import math

def fitness(input_data, net):
    error = 0.0
    outputs = []

    logic = input_data[0]
    net.reset_values()
    net.input(logic)
    net.concurrent_activation()
    error += math.fabs(net.output()[0])
    outputs.append(net.output()[0])

    logic = input_data[1]
    net.reset_values()
    net.input(logic)
    net.concurrent_activation()
    error += math.fabs(1 - net.output()[0])
    outputs.append(net.output()[0])

    logic = input_data[2]
    net.reset_values()
    net.input(logic)
    net.concurrent_activation()
    error += math.fabs(1 - net.output()[0])
    outputs.append(net.output()[0])

    logic = input_data[3]
    net.input(logic)
    net.concurrent_activation()
    error += math.fabs(net.output()[0])
    outputs.append(net.output()[0])

    #print('{:f}, {:f}, {:f}, {:f}'.format(outputs[0], outputs[1], outputs[2], outputs[3]))
    return math.pow(4 - error, 2)

params = open('py-hyperneat/tests/config_files/testConfigXor.json', 'r')
substrate_encoding = open('py-hyperneat/tests/config_files/xor_hyperneat_substrate.json', 'r')
genome_encoding = open('py-hyperneat/tests/config_files/testGenome.json', 'r')

hyper = Hyperneat()
net = NeuralNetwork([], [], 0, 0)

hyper.ea = Neat(fitness_eval=fitness, train_task=TrainTask.PREDICTION)
hyper.ea.import_config(params, genome_encoding)
#neat_ea.pop.activation_set.set_atemporal_set()
hyper.ea.pop.activation_set.unset_lin_group()
#neat_ea.pop.activation_set.use_only_sigmoid()
#hyper.ea.pop.activation_set.use_only_tanh()
hyper.import_config(substrate_encoding)
hyper.substrate.output_nodes[0].function = ActivationFunction().get('UNSIGNED_SIGMOID')
hyper.substrate.extend_nodes_list()

xor_eval = [[0, 0], [0, 1], [1, 0], [1, 1]]

for i in range(hyper.ea.max_generation):
	hyper.ea.avg_fitness = 0.0
	hyper.ea.best_epoch_fitness = 0.0

	for org in hyper.ea.pop.organisms:
		hyper.build_substrate(org, net)

		org.fitness = hyper.ea.fitness_eval(xor_eval, net)
		hyper.ea.avg_fitness += org.fitness 
		if org.fitness > hyper.ea.best_epoch_fitness:
			hyper.ea.best_epoch_fitness = org.fitness

	hyper.ea.avg_fitness /= hyper.ea.pop.params.population_max
	hyper.ea.epoch(print_stats=True)