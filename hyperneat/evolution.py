from neat.population import Population
from neat.neural_network import NeuralNetwork, CTRNN, Neuron, Connection
from neat.genome import Genome
from neat.evolution import Neat, TrainTask
from hyperneat.substrate import Substrate
from hyperneat.spatial_node import SpatialNode, SpatialNodeType

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

		# For modular CTRNN
		self.max_bias = 0.0
		self.max_delay = 0.0

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
	
	# TODO: Inheritance
	def build_modular_substrate(self, organism, substrate_set, intra_conn_table, inter_conn_table):
		# Instance of modular Continuous Recurrent Neural Network
		net = CTRNN([], [], 0, 0)

		# Total amount of input, output and total neurons
		num_inputs = 0
		num_outputs = 0
		neuron_cnt = 0

		# Map of the substrate nodes to the CTRNN
		node_gene_map = {}

		# Get CPPN network
		cppn = organism.build_phenotype()

		for idx, s in enumerate(substrate_set):
			num_inputs += s.input_count
			num_outputs += s.output_count

			for idy, n in enumerate(s.nodes):
				# Mapping by substrate pos, node pos in the respective substrate
				sn_id = (idx, idy)
				node_gene_map[sn_id] = neuron_cnt

				# Get delay and bias
				cppn_input_data = np.zeros(8) # cppn.num_inputs
				x1, y1 = n.coordinates

				# Just first four inputs set, the rest is zero
				cppn_input_data[0] = s.coordinates[0]
				cppn_input_data[1] = s.coordinates[1]
				cppn_input_data[2] = x1
				cppn_input_data[3] = y1

				cppn.reset_values()
				cppn.input(cppn_input_data)
				cppn.concurrent_activation()

				# Neuron delya and bias parameters for CTRNN activation
				delay = np.fabs(cppn.output()[1]) * self.max_delay
				if delay < 0.1: # TODO: redefine output CPPN limits
					delay = 0.1
				bias = cppn.output()[2] * self.max_bias
				
				new_neuron = Neuron(n.function, max_output=3.0)
				new_neuron.delay = delay
				new_neuron.bias = bias

				net.neurons.append(new_neuron)

				# Register the id of input and output neurons
				if n.node_type == SpatialNodeType.INPUT:
					net.in_neurons.append(neuron_cnt)
				
				if n.node_type == SpatialNodeType.OUTPUT:
					net.neurons[neuron_cnt].max_output = np.deg2rad(60)
					net.out_neurons.append(neuron_cnt)
				
				neuron_cnt += 1

			# Assuming every substrate module is equal
			for c in intra_conn_table[idx]:
				cppn_input_data = np.zeros(8) # cppn.num_inputs
				x1, y1 = s.nodes[c[0]].coordinates
				x2, y2 = s.nodes[c[1]].coordinates

				cppn_input_data[0] = s.coordinates[0]
				cppn_input_data[1] = s.coordinates[1]
				cppn_input_data[2] = x1
				cppn_input_data[3] = y1
				cppn_input_data[4] = s.coordinates[0]
				cppn_input_data[5] = s.coordinates[1]
				cppn_input_data[6] = x2
				cppn_input_data[7] = y2

				cppn.reset_values()
				cppn.input(cppn_input_data)
				cppn.concurrent_activation()

				# Intra substrate connection weight
				w = cppn.output()[0] * self.max_connection_weight

				if math.fabs(w) > self.connection_threshold:
					source = node_gene_map[(idx, c[0])]
					target = node_gene_map[(idx, c[1])]
					net.connections.append(Connection(source, target, w))

		# Compute inter substrate connections
		for c in inter_conn_table:
			cppn_input_data = np.zeros(8) # cppn.num_inputs

			# Source and target substrate
			s_substrate, t_substrate = substrate_set[c[0]], substrate_set[c[2]]
			
			xm1, ym1 = s_substrate.coordinates
			x1, y1 = s_substrate.nodes[c[1]].coordinates
			xm2, ym2 = t_substrate.coordinates
			x2, y2 = t_substrate.nodes[c[3]].coordinates

			cppn_input_data[0] = xm1
			cppn_input_data[1] = ym1
			cppn_input_data[2] = x1
			cppn_input_data[3] = y1
			cppn_input_data[4] = xm2
			cppn_input_data[5] = ym2
			cppn_input_data[6] = x2
			cppn_input_data[7] = y2

			cppn.reset_values()
			cppn.input(cppn_input_data)
			cppn.concurrent_activation()

			# Inter substrate connection weight
			w = cppn.output()[0] * self.max_connection_weight

			if math.fabs(w) > self.connection_threshold:
				source = node_gene_map[(c[0], c[1])]
				target = node_gene_map[(c[2], c[3])]
				net.connections.append(Connection(source, target, w))

		net.num_inputs = num_inputs
		net.num_outputs = num_outputs

		return net
