from population import Population
from neural_network import NeuralNetwork
from genome import Genome
import math
import sys

def fitness(org):
    error = 0.0

    net = org.build_phenotype()

    logic = [0, 0]
    net.input(logic)
    net.recursive_activation(net.num_inputs)
    error += math.fabs(net.output()[0])

    logic = [0, 1]
    net.input(logic)
    net.recursive_activation(net.num_inputs)
    error += math.fabs(1 - net.output()[0])

    logic = [1, 0]
    net.input(logic)
    net.recursive_activation(net.num_inputs)
    error += math.fabs(1 - net.output()[0])

    logic = [1, 1]
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

    print('Species = {:d}'.format(len(pop.current_species)))
    for sp in pop.current_species:

        print('Organisms = {:d}'.format(len(sp.organisms)))
        for org in sp.organisms:
            try:
                org.fitness = fitness(org)
            except:
                print(org)
                sys.exit()
    
    pop.epoch()
    for sp in pop.current_species:
        avg_fitness += sp.avg_fitness
    print('Generation #{:d} average fitness {:f}'.format(i, (avg_fitness / len(pop.current_species))))