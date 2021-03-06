import numpy as np
import math
import json
from neat.activation_functions import ActivationFunction

class Neuron:

    def __init__(self, function, layer=-1, delay=0.0, bias=0.0, max_output=0.0):
        self.input = 0.0
        self.output = 0.0
        self.activated = False
        self.incoming = False
        self.function = function
        self.layer = layer

        # For CTRNN
        self.delay = delay
        self.bias = bias

        self.max_output = max_output

    def activate(self):
        self.output = self.function(self.input)
        return self.output

class Connection:

    def __init__(self, source_id, target_id, weight, source_layer=-1, target_layer=-1):
        self.source_id = source_id
        self.target_id = target_id
        self.source_layer = source_layer
        self.target_layer = target_layer
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
        for i in range(self.num_inputs):
            self.neurons[i].activate()
        for conn in self.connections:
            self.neurons[conn.target_id].input += self.neurons[conn.source_id].output * conn.weight
        for i in range(self.num_inputs, len(self.neurons)):
            self.neurons[i].activate()
            self.neurons[i].input = 0.0
    
    def activate_multistep(self, steps):
        for _ in range(steps):
            self.activate_net()
    
    def concurrent_activation(self):
        iterations = 2 * (len(self.neurons) - (self.num_inputs + self.num_outputs)) - 1
        if iterations < 1:
            iterations = 1
        self.activate_multistep(iterations)

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
            print('Neuron' + str(i) + '_input: ' + str(self.neurons[i].input) + ' - output: ' + str(self.neurons[i].output))

#TODO: improve intialization
class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.signals = np.zeros(len(self.neurons))
        self.outputs = np.zeros(len(self.neurons))
    
    def activate(self):
        if len(self.neurons) != self.signals.size:
            raise Exception('Neuron and input signal size mismatch')

        for neuron, signal in zip(self.neurons, self.signals):
            neuron.input = signal
            neuron.activate()

    def get_activations(self):
        self.outputs = np.array([neuron.output for neuron in self.neurons])
        return self.outputs

# TODO: improve inheritance
class LayeredNetwork(NeuralNetwork):

    def __init__(self, neurons, connections, num_inputs, num_outputs, layers):
        super().__init__(neurons, connections, num_inputs, num_outputs)

        self.layers = [Layer(neurons) for neurons in layers]
        self.layer_map = {}

        for conn in self.connections:
            connection_map = (conn.source_layer, conn.target_layer)
            if connection_map not in self.layer_map:
                self.layer_map[connection_map] = np.zeros((len(self.layers[connection_map[0]].neurons), len(self.layers[connection_map[1]].neurons)))
            self.layer_map[connection_map][conn.source_id][conn.target_id] = conn.weight

    def activate_net(self):
        for key in self.layer_map:
            self.layers[key[0]].activate()            
            self.layers[key[1]].signals += np.matmul(self.layer_map[key].T, self.layers[key[0]].get_activations())

    def input(self, input_data):
        if len(input_data) != len(self.layers[0].signals):
            raise Exception('Input data and input layer neurons dimensions mismatch')

        self.layers[0].signals = np.array(input_data)

    def output(self):
        self.layers[-1].activate()
        return self.layers[-1].get_activations()

class CTRNN(NeuralNetwork):

    def __init__(self, neurons, connections, num_inputs, num_outputs):
        super().__init__(neurons, connections, num_inputs, num_outputs)

        # ID for input and output in the neurons array
        self.in_neurons = []
        self.out_neurons = []

    def input(self, input_data):
        for i, idx in enumerate(self.in_neurons):
            self.neurons[idx].input = input_data[i]

    def output(self):
        output_data = np.zeros(len(self.out_neurons))
        for i, idx in enumerate(self.out_neurons):
            neuron = self.neurons[idx]
            output_data[i] = neuron.function(neuron.output)
        return output_data

    def activate_net(self, step_time):
        """
        # First input neurons adding connections to input signal. 
        for idx, neuron in enumerate(self.neurons[:self.num_inputs]):
            neuron.input += np.sum([conn.signal for conn in self.connections if conn.target_id == idx])
            neuron_activation = neuron.function(neuron.input + neuron.bias)
            neuron.output += step_time / neuron.delay * (-neuron.output + neuron_activation)

        # Now iterate over the rest of neurons   
        for idx, neuron in enumerate(self.neurons[self.num_inputs:len(self.neurons)]):
            neuron.input = np.sum([conn.signal for conn in self.connections if conn.target_id == idx])
            neuron_activation = neuron.function(neuron.input + neuron.bias)
            neuron.output += step_time / neuron.delay * (-neuron.output + neuron_activation)
            neuron.input = 0.0 # This should solve the previous loop
        """
        for idx, neuron in enumerate(self.neurons):
            if idx in self.in_neurons:
                neuron.input += np.sum([conn.signal for conn in self.connections if conn.target_id == idx])
            else:
                neuron.input = np.sum([conn.signal for conn in self.connections if conn.target_id == idx])
            #neuron_activation = neuron.function(neuron.input + neuron.bias)
            #neuron.output += step_time / neuron.delay * (-neuron.output + neuron_activation)
            neuron.output += step_time / neuron.delay * (-neuron.output + neuron.input + neuron.bias)

            if np.fabs(neuron.output) > neuron.max_output:
                neuron.output = np.sign(neuron.output) * neuron.max_output
            #neuron.input = 0.0

        # Update connections signals for the next step
        for conn in self.connections:
            neuron = self.neurons[conn.source_id]
            conn.signal = neuron.function(neuron.output) * conn.weight
