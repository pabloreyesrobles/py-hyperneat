from neat.activation_functions import ActivationFunction

import random

class NodeType:
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2
    BIAS = 3

class ConnectionGene:

    def __init__(self, innovation = -1, incoming = -1, outgoing = -1, weight = 0.0, enable = False, source_layer = -1, target_layer = -1):
        self.innovation = innovation
        self.incoming = incoming
        self.outgoing = outgoing
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.weight = weight
        self.enable = enable
        self.mutated = False

    def randomize_weight(self, min_val=-1.0, max_val=1.0):
        self.weight = random.uniform(min_val, max_val)

    def __eq__(self, connection_gene):
        return self.innovation == connection_gene.innovation

class NodeGene:

    def __init__(self, gene_id, node_type, function, layer = -1):
        self.gene_id = gene_id
        self.node_type = node_type
        self.function = function
        self.layer = layer

    def randomize_function(self, activation_set=None):
        if activation_set == None:
            self.function = ActivationFunction().get_random_function()
        else:
            self.function = activation_set.get_random_function()

    def __eq__(self, node_gene):
        return self.gene_id == node_gene.gene_id

