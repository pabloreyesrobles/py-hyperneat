from species import Species
from genome import Genome
from genes import ConnectionGene, NodeGene, NodeType
from activation_functions import ActivationFunction
from neural_network import Neuron, Connection, NeuralNetwork
from population import Population, NodeHistoryStruct

import json
import copy
import random
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class NeatParams:
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
class NeatProb:
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

class Neat:

	def __init__(self):
		self.params = NeatParams()
		self.prob = NeatProb()
		self.pop = Population()

		self.current_generation = 0
		self.best_historical_fitness = 0.0
		
		# Must be set True to operate
		self.configurated = False

	def import_config(self, file):
		try:
			config = json.load(file)
		except ValueError:
			print('Invalid config file')
			return False

		self.params.population_max = config['populationMax']
		self.params.generations = config['generations']
		self.params.distance_coeff_1 = config['distanceCoeff1']
		self.params.distance_coeff_2 = config['distanceCoeff2']
		self.params.distance_coeff_3 = config['distanceCoeff3']
		self.params.distance_threshold = config['distanceThreshold']
		self.params.small_genome_coeff = config['smallGenomeCoeff']
		self.params.no_crossover_offspring = config['percentageOffspringWithoutCrossover']

		self.params.survival_selection = config['survivalSelection']
		self.params.allow_clones = config['allowClones']
		self.params.survival_threshold = config['survivalThreshold']
		self.params.elite_offspring_param = config['eliteOffspringParam']

		# Interspecies probability of mate
		self.prob.interspecies_mating = config['probInterspeciesMating']

		# Small population mutation probabilities
		self.prob.sp_new_node = config['sp_probAddingNewNode']
		self.prob.sp_new_connection = config['sp_probAddingNewConnection']

		# Large population mutation probabilities
		self.prob.lp_new_node = config['lp_probAddingNewNode']
		self.prob.lp_new_connection = config['lp_probAddingNewConnection']

		# Probability to change enable status of connections
		self.prob.mutate_connection_status = config['probMutateEnableConnection']

		# Mutation probabilities
		self.prob.mutation_weight = config['probChangeWeight']
		self.prob.mutate_activation = config['probChangeNodeFunction']

		self.configurated = True

	def start_generation(self, base_genome, neat_params):
		if self.import_config(neat_params) is False:
			raise NameError('Cant config population')
		#TODO: Clean genotype counts
		genome = Genome()
		if genome.import_genome(base_genome) is False:
			raise NameError('Cant load base genome')

		for node in genome.node_list:
			if self.pop.global_node_count < node.gene_id:
				self.pop.global_node_count = node.gene_id

		for connection in genome.connection_list:
			self.pop.conn_innovation_history[(connection.incoming, connection.outgoing)] = connection.innovation

			if self.pop.global_innovation_count < connection.innovation:
				self.pop.global_innovation_count = connection.innovation

		self.pop.organisms = [copy.deepcopy(genome) for i in range(self.params.population_max)]

		for org in self.organisms:
			org.randomize_weights()

		self.champion_fitness = 0.0
		self.speciate()

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

	def speciate(self):
		for sp in self.pop.species.values():
			sp.organisms = []

		for org in self.pop.organisms:
			compatible_species = False

			if len(self.pop.species) == 0:
				new_species = Species()
				new_species.birth = self.pop.get_new_species_id()
				new_species.organisms.append(org)
				
				new_species.champion_genome = org
				org.parent_species = new_species.birth # TODO: a method to assign and post-update new count

				self.pop.species[new_species.birth] = new_species
			else:
				if org.parent_species != -1 and org.parent_species in self.pop.species:
					if self.compatibility(org, self.pop.species[org.parent_species].champion_genome) < self.params.distance_threshold:
						self.pop.species[org.parent_species].organisms.append(org)
						continue

				for sp in self.pop.species.values():
					if self.compatibility(org, sp.champion_genome) < self.params.distance_threshold:
						compatible_species = True
						org.parent_species = sp.champion_genome.parent_species
						sp.organisms.append(org)
						break
				
				if compatible_species is False:
					new_species = Species()
					new_species.birth = self.pop.get_new_species_id()
					new_species.organisms.append(org)
					
					new_species.champion_genome = org
					org.parent_species = new_species.birth # TODO: a method to assign and post-update new count

					self.pop.species[new_species.birth] = new_species
	
	def mutate_add_node(self, organism):
		if len(organism.connection_list) == 0:
			return
		
		# Select connection to replace
		connection_replace = random.choice(organism.connection_list)

		if connection_replace.innovation < 0 or connection_replace.enable == False:
			return
		
		# Check if innovation already mutated
		if connection_replace.innovation in self.pop.node_history:
			new_node = self.pop.node_history[connection_replace.innovation].node_connected
			new_node.randomize_function()
			
			# Load incoming and outgoing innovation from the node_history dict
			incoming_connection_id = self.pop.node_history[connection_replace.innovation].incoming_connection_id
			outgoing_connection_id = self.pop.node_history[connection_replace.innovation].outgoing_connection_id

			# Create new connections
			incoming_connection = ConnectionGene(incoming_connection_id, connection_replace.incoming, new_node.gene_id, 1.0, True, connection_replace.source_layer, new_node.layer)
			outgoing_connection = ConnectionGene(outgoing_connection_id, new_node.gene_id, connection_replace.outgoing, 0.0, True, new_node.layer, connection_replace.target_layer)
			outgoing_connection.randomize_weight()
		else:
			new_node_layer = math.floor(abs(connection_replace.target_layer - connection_replace.source_layer) / 2) + min(connection_replace.source_layer, connection_replace.target_layer)
			if connection_replace.target_layer < new_node_layer or new_node_layer == connection_replace.source_layer or new_node_layer == connection_replace.target_layer:
				return

			new_node = NodeGene(self.pop.get_new_node_id(), NodeType.HIDDEN, ActivationFunction().get_random_function(), new_node_layer)

			# Create new connections and increment innovation
			incoming_connection = ConnectionGene(self.pop.get_new_innovation(), connection_replace.incoming, new_node.gene_id, 1.0, True, connection_replace.source_layer, new_node_layer)
			outgoing_connection = ConnectionGene(self.pop.get_new_innovation(), new_node.gene_id, connection_replace.outgoing, 0.0, True, new_node_layer, connection_replace.target_layer)
			outgoing_connection.randomize_weight()

			# Register new additions to node_history
			self.pop.node_history[connection_replace.innovation] = NodeHistoryStruct(incoming_connection.innovation, outgoing_connection.innovation, new_node)
		
		# Disable replaced connection
		connection_replace.enable = False

		organism.add_node(new_node)
		organism.add_connection(incoming_connection)
		organism.add_connection(outgoing_connection)

	def mutate_add_connection(self, organism):
		#TODO: add conditions to non-recursive and feed-forward
		input_candidate = random.choice(organism.node_list)
		source_layer_candidate = input_candidate.layer
		
		# Search for outgoing nodes
		output_candidates = []
		for node in organism.node_list:
			if node.node_type != NodeType.INPUT and node.node_type != NodeType.BIAS:
				output_candidates.append(node)
		
		if len(output_candidates) < 1:
			return

		# Select a node and verify that is not the same as input
		output_candidate = random.choice(output_candidates)
		target_layer_candidate = output_candidate.layer
		if input_candidate == output_candidate:
			return

		# Check for loops in the net
		#if self.check_loops(organism, input_candidate.gene_id, output_candidate.gene_id) is True:
		#	return
		
		# Make sure the connection doesn't exist already
		node_pair = (input_candidate.gene_id, output_candidate.gene_id)
		connection_index = -1
		for index, conn in enumerate(organism.connection_list):
			if conn.incoming == node_pair[0] and conn.outgoing == node_pair[1]:
				connection_index = index
				break

		if connection_index != -1:
			new_connection = ConnectionGene(incoming=node_pair[0], outgoing=node_pair[1], enable=True, source_layer=source_layer_candidate, target_layer=target_layer_candidate)
			
			if node_pair in self.pop.conn_innovation_history:
				new_connection.innovation = self.pop.conn_innovation_history[node_pair]
				new_connection.randomize_weight()
			else:
				# Increment global innovation before creating new ConnectionGene. #TODO make method that do this
				new_connection.innovation = self.pop.get_new_innovation()
				self.pop.conn_innovation_history[node_pair] = new_connection.innovation

			if new_connection.innovation > self.pop.global_innovation_count:
				raise NameError('Global innovation mismatched')

			organism.add_connection(new_connection)
		else:
			organism.connection_list[connection_index].randomize_weight

	def mutate_connection_weight(self, organism):
		for conn in organism.connection_list:
			if random.uniform(0, 1) < self.prob.mutation_weight:
				conn.randomize_weight()
	
	def mutate_node_functions(self, organism):
		for node in organism.node_list:
			if random.uniform(0, 1) < self.prob.mutate_activation and node.node_type != NodeType.INPUT:
				node.randomize_function()


	def mutation(self, org): 
		probabilities_set = {self.prob.lp_new_node: self.mutate_add_node(org),
							 self.prob.lp_new_connection: self.mutate_add_connection(org),
							 self.prob.mutate_activation: self.mutate_node_functions(org),
							 self.prob.mutation_weight: self.mutate_connection_weight(org)}

		probabilities_sum = np.fromiter(probabilities_set.keys(), dtype=float).sum()
		probabilities_accum = 0.0

		prob_eval = random.uniform(0, probabilities_sum)

		for prob in probabilities_set:
			probabilities_accum += prob
			if prob_eval < probabilities_accum and prob != 0.0:
				probabilities_set[prob]
				break

	# TODO:
	# Best_fitness, historical_fitness, age mechanism needed
	# champion_species tracking
	# stagnation respect to best historical fitness
	def epoch(self):

		self.pop.compute_offspring()

		self.reproduce()
		self.speciate()

		self.pop.remove_empty_species()
		
		self.current_generation += 1
	
	def reproduce(self):
		self.pop.offspring_organisms = []
		
		for sp in self.pop.species.values():
			if sp.extinct == True:
				continue

			offspring_amount = math.floor(sp.offspring)

			if offspring_amount == 0:
				continue
			
			
			elite_offspring = round(len(sp.organisms) * self.params.elite_offspring_param)
			elite_count = 0

			while offspring_amount > 0:
				if elite_count < elite_offspring:
					son = copy.deepcopy(sp.champion_genome)
					elite_count += 1
				else:
					random_mother = self.get_random_organism(sp)

					if random.uniform(0, 1) < self.params.no_crossover_offspring:
						son = copy.deepcopy(random_mother)
					else:
						if len(self.pop.species) > 1 and random.uniform(0, 1) < self.prob.interspecies_mating:
							while True:
								random_father = random.choice(list(self.pop.organisms))
								if(sp.birth != random_father.parent_species):
									break
							son = self.pop.crossover(random_mother, random_father)
						else:
							if len(sp.organisms) == 1:
								son = copy.deepcopy(random_mother)
							else:
								while True:
									random_father = self.get_random_organism(sp)
									if(random_mother is not random_father or self.params.allow_clones):
										break
								son = self.pop.crossover(random_mother, random_father)

					self.mutation(son)

				offspring_amount -= 1	
				self.pop.offspring_organisms.append(son)
				
		# Make sure the population is the correct size
		while len(self.pop.offspring_organisms) < self.params.population_max:
			random_species = random.choice(list(self.pop.species.values()))

			random_org = copy.deepcopy(random_species.champion_genome)
			random_org.randomize_weights()

			random_species.organisms.append(random_org)
			self.pop.offspring_organisms.append(random_org)
		
		self.pop.organisms = self.pop.offspring_organisms

	def get_random_organism(self, species):
		if species.extinct == True or len(species.organisms) == 0:
			return None

		survival_cap = math.floor(self.params.survival_threshold * len(species.organisms))

		if survival_cap == 0:
			return species.organisms[0]
		else:
			return random.choice(species.organisms[:survival_cap])


		