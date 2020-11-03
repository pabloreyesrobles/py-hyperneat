from hyperneat.spatial_node import SpatialNode, SpatialNodeType
from neat.activation_functions import ActivationFunction
import json

class SubstrateType:
	SUBSTRATE_2D = 0
	SUBSTRATE_3D = 1
	SUBSTRATE_RADIAL = 2
	SUBSTRATE_STATE_SPACE = 3

class Substrate:

	def __init__(self):
		self.nodes = []

		self.input_nodes = []
		self.hidden_nodes = []
		self.output_nodes = []

		self.input_count = 0
		self.hidden_count = 0
		self.output_count = 0		

		self.num_layers = 0

		self.global_node_count = 0

		self.activation_function = None

	def import_substrate(self, data):
		layers = []
		layer_count = 0
		self.num_layers = len(data['Layers'])
		self.activation_function = ActivationFunction().get(data['nodeFunction'])

		for layer in data['Layers']:
			temp_nodes = []
			for node in layer['nodesInfo']:
				node_type = node[0]
				node_id = node[1]
				coordinates = [node[2], node[3]]

				node_obj = SpatialNode(node_id, node_type, coordinates, self.activation_function, layer_count)

				temp_nodes.append(node_obj)

				if node_type == SpatialNodeType.INPUT or node_type == SpatialNodeType.BIAS:
					self.input_nodes.append(node_obj)
				elif node_type == SpatialNodeType.HIDDEN:
					self.hidden_nodes.append(node_obj)
				else:
					self.output_nodes.append(node_obj)
			
			layer_count += 1
			layers.append(temp_nodes)
		
		self.input_count = len(self.input_nodes)
		self.hidden_count = len(self.hidden_nodes)
		self.output_count = len(self.output_nodes)

		self.extend_nodes_list()

	def extend_nodes_list(self):
		self.nodes = []

		# First add input and output nodes to self.nodes array
		self.nodes.extend(self.input_nodes)
		self.nodes.extend(self.hidden_nodes)
		self.nodes.extend(self.output_nodes)




		