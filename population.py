from species import Species
from genome import Genome
from genes import ConnectionGene, NodeGene, NodeType
from activation_functions import ActivationFunction
from neural_network import Neuron, Connection, NeuralNetwork

import json
import copy
import random
from dataclasses import dataclass

@dataclass
class PopulationParams:
	population_max: int = 0
	generations: int = 0
	distance_coeff_1: float = 0.0
	distance_coeff_2: float = 0.0
	distance_coeff_3: float = 0.0
	distance_threshold: float = 0.0
	small_genome_coeff: float = 0.0
	no_crossover_offspring: float = 0.0
	survival_selection: bool = False
	allow_clones: bool = False
	survival_threshold: float = 0.0
	elite_offspring_param: float = 0.0

@dataclass
class PopulationProb:
	# Interspecies probability of mate
	interspecies_mating: float = 0.0

	# Small population mutation probabilities
	sp_new_node: float = 0.0
	sp_new_connection: float = 0.0

	# Large population mutation probabilities
	lp_new_node: float = 0.0
	lp_new_connection: float = 0.0

	# Probability to change enable status of connections
	mutate_connection_status: float = 0.0

	# Mutation probabilities
	mutation_weight: float = 0.0
	mutate_activation: float = 0.0

class NodeHistoryStruct:
	
	def __init__(self, incoming_connection_id, outgoing_connection_id, node_connected):
		self.incoming_connection_id = incoming_connection_id
		self.outgoing_connection_id = outgoing_connection_id
		self.node_connected = node_connected

class Population:

	def __init__(self):
		self.params = PopulationParams()
		self.prob = PopulationProb()

		self.current_generation = 0
		self.champion_fitness = 0.0
		self.global_innovation_count = 0
		self.global_node_count = 0

		self.organisms = []
		self.current_species = []

		self.conn_innovation_history = {}
		self.node_history = {}

		# Must be set True to operate
		self.configurated = False

	def import_config(self, file):
		try:
			config = json.load(file)
		except ValueError:
			print('Invalid population config file')
			return False

		self.params.population_max = config['populationMax']
		self.params.generations = config['generations']
		self.params.distance_coeff_1 = config['distanceCoeff1']
		self.params.distance_coeff_2 = config['distanceCoeff2']
		self.params.distance_coeff_3 = config['distanceCoeff3']
		self.params.distance_threshold = config['distanceThreshold']
		self.params.small_genome_coeff = config['smallGenomeCoeff']
		self.params.no_crossover_offspring = config['percentageOffspringWithoutCrossover']

		self.params.survival_selection = config['probInterspeciesMating']
		self.params.allow_clones = config['sp_probAddingNewNode']
		self.params.survival_threshold = config['sp_probAddingNewConnection']
		self.params.elite_offspring_param = config['lp_probAddingNewNode']

		# Interspecies probability of mate
		self.prob.interspecies_mating = config['lp_probAddingNewConnection']

		# Small population mutation probabilities
		self.prob.sp_new_node = config['probMutateEnableConnection']
		self.prob.sp_new_connection = config['probChangeWeight']

		# Large population mutation probabilities
		self.prob.lp_new_node = config['probChangeNodeFunction']
		self.prob.lp_new_connection = config['survivalSelection']

		# Probability to change enable status of connections
		self.prob.mutate_connection_status = config['allowClones']

		# Mutation probabilities
		self.prob.mutation_weight = config['survivalThreshold']
		self.prob.mutate_activation = config['eliteOffspringParam']

		self.configurated = True

	def get_new_innovation(self):
		self.global_innovation_count += 1

		return self.global_innovation_count

	def get_new_node_id(self):
		self.global_node_count += 1

		return self.global_node_count

	def start_generation(self, base_genome, pop_params):
		if self.import_config(pop_params) is False:
			raise NameError('Cant config population')
		#TODO: Clean genotype counts
		genome = Genome()
		if genome.import_genome(base_genome) is False:
			raise NameError('Cant load base genome')

		for node in genome.node_list:
			if self.global_node_count < node.gene_id:
				self.global_node_count = node.gene_id

		for connection in genome.connection_list:
			self.conn_innovation_history[(connection.incoming, connection.outgoing)] = connection.innovation

			if self.global_innovation_count < connection.innovation:
				self.global_innovation_count = connection.innovation

		self.organisms = [copy.deepcopy(genome) for i in range(self.params.population_max)]

		for org in self.organisms:
			org.randomize_weights()

		self.champion_fitness = 0.0
		self.speciate(self.organisms, self.current_species)

	def compatibility(self, org_A, org_B):
		excess = 0
		disjoint = 0
		var_weight = 0

		itr_A = 0
		itr_B = 0

		conn_A = org_A.connection_list[itr_A]
		conn_B = org_B.connection_list[itr_B]

		larger_genome = max(len(org_A.connection_list), len(org_B.connection_list))

		while True:
			if conn_A.innovation > conn_B.innovation:
				disjoint += 1
				itr_B += 1
			elif conn_A.innovation == conn_B.innovation:
				var_weight += abs(conn_A.weight - conn_B.weight)
				itr_A += 1
				itr_B += 1
			else:
				disjoint += 1
				itr_B += 1

			if itr_A >= len(org_A.connection_list):
				while itr_B < len(org_B.connection_list):
					excess += 1
					itr_B += 1
				break
			
			if itr_B >= len(org_B.connection_list):
				while itr_A < len(org_A.connection_list):
					excess += 1
					itr_A += 1
				break

			conn_A = org_A.connection_list[itr_A]
			conn_B = org_B.connection_list[itr_B]
		
		if larger_genome > self.params.small_genome_coeff:
			divisor = larger_genome
			small_genomes_buff = 1.0
		else:
			divisor = 1.0
			if self.params.distance_coeff_3 < 1.0:
				small_genomes_buff = 1.0 / self.params.distance_coeff_3
			else:
				small_genomes_buff = 1.0
		
		return self.params.distance_coeff_1 * excess / divisor + self.params.distance_coeff_2 * disjoint / divisor + self.params.distance_coeff_3 * var_weight * small_genomes_buff

	def speciate(self, organism_list, species_list):
		for spec in species_list:
			spec.organisms = []

		for org in organism_list:
			compatible_species = False

			if len(species_list) == 0:
				new_species = Species()
				new_species.organisms.append(org)
				new_species.representant = org

				species_list.append(new_species)
			else:
				for spec in species_list:
					if self.compatibility(org, spec.representant) < self.params.distance_threshold:
						compatible_species = True
						spec.organisms.append(org)
						break
				
				if compatible_species is False:
					new_species = Species()
					new_species.organisms.append(org)
					new_species.representant = org

					species_list.append(new_species)
	
	def mutate_add_node(self, organism):
		if len(organism.connection_list) == 0:
			return
		
		# Select connection to replace
		connection_id = random.randint(0, len(organism.connection_list) - 1)
		connection_replace = organism.connection_list[connection_id]

		if connection_replace.innovation < 0 or connection_replace.enable == False:
			return

		# Check if innovation already mutated
		if connection_replace.innovation in self.node_history:
			new_node = self.node_history[connection_replace.innovation].node_connected
			new_node.randomize_function()
			
			# Load incoming and outgoing innovation from the node_history dict
			incoming_connection_id = self.node_history[connection_replace.innovation].incoming_connection_id
			outgoing_connection_id = self.node_history[connection_replace.innovation].outgoing_connection_id

			# Create new connections
			incoming_connection = ConnectionGene(incoming_connection_id, connection_replace.incoming, new_node.gene_id, 1.0, True)
			outgoing_connection = ConnectionGene(outgoing_connection_id, new_node.gene_id, connection_replace.outgoing, random.uniform(-1.0, 1.0), True)
		else:
			new_node = NodeGene(self.get_new_node_id(), NodeType.HIDDEN, ActivationFunction().get_random_function())

			# Create new connections and increment innovation
			incoming_connection = ConnectionGene(self.get_new_innovation(), connection_replace.incoming, new_node.gene_id, 1.0, True)
			outgoing_connection = ConnectionGene(self.get_new_innovation(), new_node.gene_id, connection_replace.outgoing, random.uniform(-1.0, 1.0), True)

			# Register new additions to node_history
			self.node_history[connection_replace.innovation] = NodeHistoryStruct(incoming_connection.innovation, outgoing_connection.innovation, new_node)

		connection_replace.enable = False

		organism.add_node(new_node)
		organism.add_connection(incoming_connection)
		organism.add_connection(outgoing_connection)

	def mutate_add_connection(self, organism):
		#TODO: add conditions to non-recursive and feed-forward
		input_candidate = random.choice(organism.node_list)
		
		# Search for outgoing nodes
		output_candidates = []
		for node in organism.node_list:
			if node.node_type != NodeType.INPUT and node.node_type != NodeType.BIAS:
				output_candidates.append(node)
		
		if len(output_candidates) < 1:
			return

		# Select a node and verify that is not the same as input
		output_candidate = random.choice(output_candidates)
		if input_candidate == output_candidate:
			return

		# Check for loops in the net
		if self.check_loops(organism, input_candidate, output_candidate) is True:
			return
		
		# Make sure the connection doesn't exist already
		node_pair = (input_candidate.gene_id, output_candidate.gene_id)
		connection_index = -1
		for index, conn in enumerate(organism.connection_list):
			if conn.incoming == node_pair[0] and conn.outgoing == node_pair[1]:
				connection_index = index
				break

		if connection_index != -1:			
			new_connection = ConnectionGene(incoming=node_pair[0], outgoing=node_pair[1], enable=True)
			
			if node_pair in self.conn_innovation_history:
				new_connection.innovation = self.conn_innovation_history[node_pair]
				new_connection.randomize_weight()
			else:
				# Increment global innovation before creating new ConnectionGene. #TODO make method that do this
				new_connection.innovation = self.get_new_innovation()
				self.conn_innovation_history[node_pair] = new_connection.innovation

			if new_connection.innovation > self.global_innovation_count:
				raise NameError('Global innovation mismatched')

			organism.add_connection(new_connection)
		else:
			organism.connection_list[connection_index].randomize_weight

	def check_loops(self, organism, inspect, itr):
		for conn in organism.connection_list:
			if conn.incoming == itr:
				if conn.outgoing == inspect:
					return True
				else:
					self.check_loops(organism, inspect, conn.outgoing)
		
		return False