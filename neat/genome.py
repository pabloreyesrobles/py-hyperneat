from neat.genes import ConnectionGene, NodeGene, NodeType
from neat.neural_network import Neuron, Connection, NeuralNetwork, LayeredNetwork
from neat.activation_functions import ActivationFunction

import json
import random
import copy
import math

class Phenotype:
    NONE = 0
    LAYERED_NETWORK = 1

class Genome:

    def __init__(self, num_layers=0, phenotype=Phenotype.NONE, weights_range=[-1.0, 1.0]):
        self.node_list = []
        self.connection_list = []
        self.complexity = 0

        self.num_inputs = 0
        self.num_outputs = 0

        self.fitness = 0.0
        self.shared_fitness = 0.0
        
        self.phenotype = phenotype
        self.num_layers = num_layers

        self.parent_species = -1
        self.weights_range = weights_range

    def add_node(self, node_gene):
        for node in self.node_list:
            # Node already exist
            if node.gene_id == node_gene.gene_id:
                #self.node_list[pos].randomize_function()
                return 1

        if node_gene.node_type == NodeType.INPUT:
            self.num_inputs += 1
        elif node_gene.node_type == NodeType.OUTPUT:
            self.num_outputs += 1

        self.complexity += 1
        self.node_list.append(node_gene)
        return 0

    def add_connection(self, connection_gene):
        for pos, val in enumerate(self.connection_list):
            if val.innovation == connection_gene.innovation:
                self.connection_list[pos] = connection_gene
                self.connection_list[pos].mutated = False
                return 1
        
        self.connection_list.append(connection_gene)
        return 0

    def create_genome_by_size(self, input_size, output_size):
        node_id_cnt = 0
        innovation_cnt = 0
        input_id_arr = []
        output_id_arr = []

        # Create node genes
        while input_size > 0:
            new_node = NodeGene(node_id_cnt, NodeType.INPUT, ActivationFunction().get('TANH'), 0)
            self.add_node(new_node)
            input_id_arr.append(node_id_cnt)
            node_id_cnt += 1
            input_size -= 1
        
        while output_size > 0:
            self.node_list.append(NodeGene(node_id_cnt, NodeType.OUTPUT, ActivationFunction().get('TANH'), self.num_layers - 1))
            output_id_arr.append(node_id_cnt)
            node_id_cnt += 1
            output_size -= 1

        # Create connection genes
        for i in input_id_arr:
            for o in output_id_arr:
                self.connection_list.append(ConnectionGene(innovation_cnt, i, o, 0.0, True, self.node_list[i].layer, self.node_list[o].layer))
                innovation_cnt += 1

        self.randomize_weights()

    def create_genome_from_genes(self, nodes, connections, phenotype=Phenotype.NONE, num_layers=-1):
        for node_gene in nodes:
            self.add_node(node_gene)

        for connection_gene in connections:
            self.add_connection(connection_gene)

        #TODO: fix phenotype construction
        if num_layers == -1:
            raise Exception('Number of layers missing')

        self.phenotype = phenotype
        self.num_layers = num_layers
            
    def import_genome(self, file):
        try:
            data = json.load(file)
        except ValueError:
            print('Invalid genome config file')
            return False

        for neuron in data['GeneticEncoding']['nodes']:
            node_gene = NodeGene(neuron['nodeID'], neuron['type'], ActivationFunction().get(neuron['function']), neuron['row'])
            self.add_node(node_gene)

        for connection in data['GeneticEncoding']['connections']:
            source_layer = self.node_list[connection['in']].layer
            target_layer = self.node_list[connection['out']].layer
            connection_gene = ConnectionGene(connection['innovation'], connection['in'], connection['out'], connection['weight'], connection['enable'], source_layer, target_layer)
            self.add_connection(connection_gene)
        
        self.phenotype = data['phenotype']
        if self.phenotype == Phenotype.LAYERED_NETWORK:
            self.num_layers = data['num_layers']

        if 'weight_range' in data:
            self.weights_range = data['weight_range']

        return True

    def export_genome(self):
        data = {}
        data['phenotype'] = self.phenotype
        data['num_layers'] = self.num_layers

        data['GeneticEncoding'] = {}
        data['GeneticEncoding']['nodes'] = []
        data['GeneticEncoding']['connections'] = []

        for node_gene in self.node_list:
            data['GeneticEncoding']['nodes'].append({
                'exist': 'true',
                'nodeID': node_gene.gene_id,
                'type': node_gene.node_type,
                'row': node_gene.layer,
                'function': ActivationFunction().get_function_name(node_gene.function)
            })

        for connection_gene in self.connection_list:
            data['GeneticEncoding']['connections'].append({
                'exist': 'true',
                'innovation': connection_gene.innovation,
                'in': connection_gene.incoming,
                'out': connection_gene.outgoing,
                'weight': connection_gene.weight,
                'enable': connection_gene.enable
            })

        return data

    def save_genome(self):
        data = self.export_genome()
        with open('champion.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)

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
    
    def recompute_complexity(self):
        self.complexity = len(self.node_list)

    def randomize_weights(self):
        for conn in self.connection_list:
            conn.randomize_weight(self.weights_range[0], self.weights_range[1])
    
    def randomize_functions(self):
        for node in self.node_list:
            node.randomize_function()

    def new_node_layer(self, source_layer, target_layer):
        return math.floor(abs(target_layer - source_layer) / 2) + min(source_layer, target_layer)

    # TODO: merge it with build_phenotype depending on self.phenotype
    def build_layered_phenotype(self):
        layers = [[] for i in range(self.num_layers)]
        conn_map = {}

        neurons = []
        for node_gene in self.node_list:
            conn_map[node_gene.gene_id] = len(layers[node_gene.layer])
            neuron = Neuron(node_gene.function, node_gene.layer)

            neurons.append(neuron)
            layers[node_gene.layer].append(neuron)
        
        connections = []
        for connection_gene in self.connection_list:
            if connection_gene.innovation == -1 or connection_gene.enable == False:
                continue
            connection = Connection(conn_map[connection_gene.incoming],
                                    conn_map[connection_gene.outgoing],
                                    connection_gene.weight,
                                    connection_gene.source_layer,
                                    connection_gene.target_layer)
            connections.append(connection)
        
        return LayeredNetwork(neurons, connections, self.num_inputs, self.num_outputs, layers)
