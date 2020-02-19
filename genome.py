from genes import ConnectionGene, NodeGene, NodeType
from neural_network import Neuron, Connection, NeuralNetwork
from activation_functions import ActivationFunction

import json
import random
import copy

class Genome:

    def __init__(self):
        self.node_list = []
        self.connection_list = []

        self.num_inputs = 0
        self.num_outputs = 0

        self.fitness = 0.0
        self.shared_fitness = 0.0

    def add_node(self, node_gene):
        for pos, val in enumerate(self.node_list):
            # Node already exist
            if val.gene_id == node_gene.gene_id:
                self.node_list[pos].randomize_function()
                return 1

        if node_gene.node_type == NodeType.INPUT:
            self.num_inputs += 1
        elif node_gene.node_type == NodeType.OUTPUT:
            self.num_outputs += 1

        self.node_list.append(node_gene)
        return 0

    def add_connection(self, connection_gene):
        for pos, val in enumerate(self.connection_list):
            if val.innovation == connection_gene.innovation:
                self.connection_list[pos] = connection_gene
                return 1
        
        self.connection_list.append(connection_gene)
        return 0
            
    def import_genome(self, file):
        try:
            data = json.load(file)
        except ValueError:
            print('Invalid genome config file')
            return False

        for neuron in data['GeneticEncoding']['nodes']:
            node_gene = NodeGene(neuron['nodeID'], neuron['type'], ActivationFunction().get(neuron['function']))
            self.add_node(node_gene)

        for connection in data['GeneticEncoding']['connections']:
            connection_gene = ConnectionGene(connection['innovation'], connection['in'], connection['out'], connection['weight'], connection['enable'])
            self.add_connection(connection_gene)

        return True

    def build_phenotype(self):
        # Init some variables
        node_gene_map = {}
        net_neurons = []
        net_connections = []
        neuron_count = 0
        
        for node_gene in self.node_list:
            neuron = Neuron(node_gene.function)
            net_neurons.append(neuron)
            node_gene_map[node_gene.gene_id] = neuron_count
            neuron_count += 1

        for connection_gene in self.connection_list:
            if connection_gene.innovation == -1 or connection_gene.enable == False:
                continue

            connection = Connection(node_gene_map[connection_gene.incoming], node_gene_map[connection_gene.outgoing], connection_gene.weight)
            net_connections.append(connection)

        return NeuralNetwork(net_neurons, net_connections, self.num_inputs, self.num_outputs)

    def eval(self, input_data):
        net = self.build_phenotype()
        net.reset_values()
        net.input(input_data)

        for i in range(net.num_outputs):
            net.recursive_activation(net.num_inputs + i)

        return net.output

    def randomize_weights(self):
        for conn in self.connection_list:
            conn.randomize_weight()
    
    def randomize_functions(self):
        for node in self.node_list:
            node.randomize_function()
