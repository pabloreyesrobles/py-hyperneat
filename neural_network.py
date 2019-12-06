import math
import json
from activation_functions import ActivationFunction

class Neuron:

    def __init__(self, function):
        self.input = 0.0
        self.output = 0.0
        self.activated = False
        self.incoming = False
        self.function = function

    def activate(self):
        self.output = self.function(self.input)

class Connection:

    def __init__(self, source_id, target_id, weight):
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.signal = 0.0

#Fundamental network, just activation of the net
class NeuralNetwork:

    def __init__(self, neurons, connections, num_inputs, num_outputs):
        self.neurons = neurons
        self.connections = connections
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
    
    def clear(self):
        self.neurons = []
        self.connections = []
        self.num_inputs = 0
        self.num_outputs = 0
    
    def import_net(self, file):
        data = json.load(file)
        for neuron in data['GeneticEncoding']['nodes']:
            self.neurons.append(Neuron(neuron['function']))
            if neuron['type'] == 0:
                self.num_inputs = self.num_inputs + 1
            elif neuron['type'] == 2:
                self.num_outputs = self.num_outputs + 1
        for connection in data['GeneticEncoding']['connections']:
            if connection['enable'] == 1 and connection['exist'] == True:
                self.connections.append(Connection(connection['in'], connection['out'], connection['weight']))

    def reset_values(self):
        for i in range(len(self.neurons)):
            self.neurons[i].input = 0
            self.neurons[i].output = 0
            self.neurons[i].activated = False
        for i in range(len(self.connections)):
            self.connections[i].signal = 0
    
    def add_connection(self, connection):
        self.connections.append(connection)
        self.neurons[connection.target_id].incoming = True

    def activate_net(self):
        for i in range(len(self.connections)):
            self.connections[i].signal = self.neurons[self.connections[i].source_id].output * self.connections[i].weight
        for i in range(len(self.connections)):
            self.neurons[self.connections[i].target_id].input += self.connections[i].signal
        for i in range(0, len(self.neurons)):
            self.neurons[i].Activate()

    def recursive_activation(self, neuron_id):
        for conn in self.connections:
            if conn.target_id == neuron_id:
                self.recursive_activation(conn.source_id)
                conn.signal = self.neurons[conn.source_id].output * conn.weight
                self.neurons[neuron_id].input = self.neurons[neuron_id].input + conn.signal
        if not self.neurons[neuron_id].activated:
            self.neurons[neuron_id].activate()
            self.neurons[neuron_id].activated = True
                
    def input(self, input_data):
        for i in range(self.num_inputs):
            self.neurons[i].input = input_data[i]

    def output(self):
        t_output = []
        for i in range(self.num_outputs):
            t_output.append(self.neurons[i + self.num_inputs].output)
        return t_output
    
    def print_connections(self):
        for pos, val in enumerate(self.connections):
            print('Connection ' + str(pos))
            print('Source ID: {:d} - Target ID: {:d} - weight: {:f} - signal: {:f}'.format(val.source_id, val.target_id, val.weight, val.signal))
    
    def print_neurons(self):
        for i in range(len(self.neurons)):
            print('Neuron ' + str(i))
            print('input: ' + str(self.neurons[i].input) + ' - output: ' + str(self.neurons[i].output))
