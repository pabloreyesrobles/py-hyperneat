from neat.population import Population
from neat.neural_network import NeuralNetwork, Neuron, Connection
from neat.genome import Genome
from neat.evolution import Neat, TrainTask
from hyperneat.substrate import Substrate

import json
import copy
import random
import math
import numpy as np

# TODO: make CPPN inputs functions file
# vector: [x1, y1, x2, y2]
def euclidean_distance(vector):
	return math.sqrt((vector[0] + vector[2]) ** 2 + (vector[1] + vector[3]) ** 2)

class Hyperneat:

	def __init__(self):
		self.ea = Neat()
		self.substrate = Substrate()

		self.connection_threshold = 0.0
		self.max_connection_weight = 0.0

	def import_config(self, config_file):
		try:
			config = json.load(config_file)
		except ValueError:
			print('HyperNEAT: Invalid config file')
			return False

		self.connection_threshold = float(config['connectionThreshold'])
		self.max_connection_weight = float(config['maxConnectionWeight'])

		self.substrate.import_substrate(config['Substrate'])

	def build_substrate(self, organism, net):
		neuron_count = len(self.substrate.nodes)
		
		net.clear()		
		net.num_inputs = self.substrate.input_count
		net.num_outputs = self.substrate.output_count

		hidden_count = neuron_count - (net.num_inputs + net.num_outputs)
		hidden_offset = net.num_inputs + net.num_outputs

		net.neurons = [Neuron(self.substrate.activation_function) for i in range(neuron_count)]

		if hidden_count > 0:
			for i in range(net.num_inputs):
				inputs = [0.0] * 5
				inputs[0] = self.substrate.nodes[i].coordinates[0]
				inputs[1] = self.substrate.nodes[i].coordinates[1]

				for j in range(hidden_count):
					inputs[2] = self.substrate.nodes[hidden_offset + j].coordinates[0]
					inputs[3] = self.substrate.nodes[hidden_offset + j].coordinates[1]
					inputs[4] = euclidean_distance(inputs)

					outputs = organism.eval(inputs)
					weight = outputs()[0] * self.max_connection_weight

					if math.fabs(weight) > self.connection_threshold:
						connection = Connection(i, hidden_offset + j, weight)
						net.add_connection(connection)

			for i in range(hidden_count):
				inputs = [0.0] * 5
				inputs[0] = self.substrate.nodes[hidden_offset + i].coordinates[0]
				inputs[1] = self.substrate.nodes[hidden_offset + i].coordinates[1]

				for j in range(net.num_outputs):
					inputs[2] = self.substrate.nodes[net.num_inputs + j].coordinates[0]
					inputs[3] = self.substrate.nodes[net.num_inputs + j].coordinates[1]
					inputs[4] = euclidean_distance(inputs)

					outputs = organism.eval(inputs)
					weight = outputs()[0] * self.max_connection_weight

					if math.fabs(weight) > self.connection_threshold:
						connection = Connection(hidden_offset + i, net.num_inputs + j, weight)
						net.add_connection(connection)

		else:
			for i in range(net.num_inputs):
				inputs = [0.0] * 5
				inputs[0] = self.substrate.nodes[i].coordinates[0]
				inputs[1] = self.substrate.nodes[i].coordinates[1]

				for j in range(net.num_outputs):
					inputs[2] = self.substrate.nodes[net.num_inputs + j].coordinates[0]
					inputs[3] = self.substrate.nodes[net.num_inputs + j].coordinates[1]
					inputs[4] = euclidean_distance(inputs)

					outputs = organism.eval(inputs)
					weight = outputs()[0] * self.max_connection_weight

					if math.fabs(weight) > self.connection_threshold:
						connection = Connection(i, net.num_inputs + j, weight)
						net.add_connection(connection)

		return True
