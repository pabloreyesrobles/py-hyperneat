from activation_functions import ActivationFunction

class NodeType:
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2
    BIAS = 3

class ConnectionGene:

    def __init__(self, innovation = -1, incoming = -1, outgoing = -1, weight = 0.0, enable = False):
        self.innovation = innovation
        self.incoming = incoming
        self.outgoing = outgoing
        self.weight = weight
        self.enable = enable

class NodeGene:

    def __init__(self, gene_id, node_type, function):
        self.gene_id = gene_id
        self.node_type = node_type
        self.function = function

    def randomize_function(self):
        self.funcion = ActivationFunction().get_random_function()

