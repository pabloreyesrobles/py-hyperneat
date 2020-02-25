from population import Population
from neural_network import NeuralNetwork
from genome import Genome
from neat import Neat, TrainTask
import math
import sys

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

params = open('testConfig.json', 'r')
genome = open('testXor.json', 'r')

evolution = Neat(fitness_eval=fitness, train_task=TrainTask.PREDICTION)
evolution.import_config(params, genome)

xor_eval = [[0, 0], [0, 1], [1, 0], [1, 1]]

for i in range(evolution.max_generation):

    evolution.set_multi_input(xor_eval)
    evolution.evaluate_population()    
    evolution.epoch()

    print('Species = {:d}, champion_fitness: {:f}, avg_fitness: {:f}, best_fitness: {:f}'.format(len(evolution.pop.species), evolution.pop.champion_fitness, evolution.avg_fitness / evolution.pop.params.population_max, evolution.best_epoch_fitness))

evolution.pop.champion_genome.save_genome()