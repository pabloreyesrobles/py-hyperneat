from neat.activation_functions import ActivationFunction

class SpatialNodeType:
	INPUT = 0
	HIDDEN = 1
	OUTPUT = 2
	BIAS = 3

class SpatialNode:

	def __init__(self, node_id, node_type, coordinates, function, layer=-1):
		self.node_id = node_id
		self.node_type = node_type
		self.coordinates = coordinates
		self.function = function
		self.layer = layer
	
