from population import Population
from neural_network import NeuralNetwork
from genome import Genome
import math
import sys

def fitness(org):
    error = 0.0

    net = org.build_phenotype()

    logic = [0, 0]
    net.reset_values()
    net.input(logic)
    net.recursive_activation(net.num_inputs)
    error += math.fabs(net.output()[0])

    logic = [0, 1]
    net.reset_values()
    net.input(logic)
    net.recursive_activation(net.num_inputs)
    error += math.fabs(1 - net.output()[0])

    logic = [1, 0]
    net.reset_values()
    net.input(logic)
    net.recursive_activation(net.num_inputs)
    error += math.fabs(1 - net.output()[0])

    logic = [1, 1]
    net.reset_values()
    net.input(logic)
    net.recursive_activation(net.num_inputs)
    error += math.fabs(net.output()[0])

    return math.pow(4 - error, 2)

params = open('testConfig.json', 'r')
genome = open('testXor.json', 'r')

pop = Population()
pop.start_generation(genome, params)

for i in range(pop.params.generations):
    avg_fitness = 0.0

    for sp in pop.current_species:
        for org in sp.organisms:
            try:
                org.fitness = fitness(org)
                avg_fitness += org.fitness
            except:
                print(org)
                raise Exception('Error in organism')
    
    print('Species = {:d}, champion_fitness: {:f}, avg_fitness: {:f}'.format(len(pop.current_species), pop.champion_fitness, avg_fitness / pop.params.population_max))
    pop.epoch()