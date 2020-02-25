from population import Population
from neural_network import NeuralNetwork
from genome import Genome
import math
import sys

def fitness(org):
    error = 0.0
    outputs = []

    net = org.build_layered_phenotype()

    logic = [0, 0]
    net.reset_values()
    net.input(logic)
    net.activate_net()
    error += math.fabs(net.output()[0])
    outputs.append(net.output()[0])

    logic = [0, 1]
    net.reset_values()
    net.input(logic)
    net.activate_net()
    error += math.fabs(1 - net.output()[0])
    outputs.append(net.output()[0])

    logic = [1, 0]
    net.reset_values()
    net.input(logic)
    net.activate_net()
    error += math.fabs(1 - net.output()[0])
    outputs.append(net.output()[0])

    logic = [1, 1]
    net.reset_values()
    net.input(logic)
    net.activate_net()
    error += math.fabs(net.output()[0])
    outputs.append(net.output()[0])

    #print('{:f}, {:f}, {:f}, {:f}'.format(outputs[0], outputs[1], outputs[2], outputs[3]))
    return math.pow(4 - error, 2)

params = open('testConfig.json', 'r')
genome = open('testXor.json', 'r')

pop = Population()
pop.start_generation(genome, params)

for i in range(pop.params.generations):
    avg_fitness = 0.0
    best_epoch_fitness = 0.0
    count = 0
    for org in pop.organisms:
        try:
            org.fitness = fitness(org)
            avg_fitness += org.fitness
            count += 1
            if org.fitness > best_epoch_fitness:
                best_epoch_fitness = org.fitness
        except:
            print(org)
            raise Exception('Error in organism')
    
    pop.epoch()
    print('Species = {:d}, champion_fitness: {:f}, avg_fitness: {:f}, best_fitness: {:f}'.format(len(pop.species), pop.champion_fitness, avg_fitness / pop.params.population_max, best_epoch_fitness))

pop.champion_genome.save_genome()