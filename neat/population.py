from neat.species import Species
from neat.genome import Genome
from neat.genes import ConnectionGene, NodeGene, NodeType
from neat.activation_functions import ActivationFunction
from neat.neural_network import Neuron, Connection, NeuralNetwork

import json
import copy
import random
import math
import numpy as np
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
	min_species: int = 0
	max_species: int = 0
	stagnation_purge: bool = False

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
		self.champion_genome = None
		self.best_historical_fitness = 0.0

		self.global_innovation_count = 0
		self.global_node_count = 0
		self.global_species_count = -1

		self.organisms = []
		self.offspring_organisms = []
		self.species = {}

		self.conn_innovation_history = {}
		self.node_history = {}

		self.distance_threshold_var = 0.0
		self.speciation_adjust_start = False

		self.activation_set = ActivationFunction()
		# Must be set True to operate
		self.configurated = False

	def config_population(self, config):
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

		self.params.min_species = config['adjustMinSpecies']
		self.params.max_species = config['adjustMaxSpecies']

		self.params.stagnation_purge = config['stagnationPurge']

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

	def get_new_innovation(self):
		self.global_innovation_count += 1
		return self.global_innovation_count

	def get_new_node_id(self):
		self.global_node_count += 1
		return self.global_node_count

	def get_new_species_id(self):
		self.global_species_count += 1
		return self.global_species_count

	def start_population(self, genome):
		self.seed_genome = copy.deepcopy(genome)

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
		self.speciate()

	def restart_population(self):
		self.current_generation = 0
		self.champion_fitness = 0.0
		self.best_historical_fitness = 0.0

		self.global_innovation_count = 0
		self.global_node_count = 0
		self.global_species_count = -1

		self.organisms = []
		self.offspring_organisms = []
		self.species = {}

		self.conn_innovation_history = {}
		self.node_history = {}

		self.distance_threshold_var = 0.0
		self.speciation_adjust_start = False

		for node in self.seed_genome.node_list:
			if self.global_node_count < node.gene_id:
				self.global_node_count = node.gene_id

		for connection in self.seed_genome.connection_list:
			self.conn_innovation_history[(connection.incoming, connection.outgoing)] = connection.innovation
			if self.global_innovation_count < connection.innovation:
				self.global_innovation_count = connection.innovation

		self.organisms = [copy.deepcopy(self.seed_genome) for i in range(self.params.population_max)]

		for org in self.organisms:
			org.randomize_weights()

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
			small_genomes_buff = 1.0 / larger_genome
		else:
			divisor = 1
			if self.params.distance_coeff_3 < 1.0:
				small_genomes_buff = 1.0 / self.params.distance_coeff_3
			else:
				small_genomes_buff = 1.0
		
		compatibility = self.params.distance_coeff_1 * excess / divisor + self.params.distance_coeff_2 * disjoint / divisor + self.params.distance_coeff_3 * var_weight * small_genomes_buff
		return compatibility

	def sort_organisms(self):
		self.organisms.sort(key=lambda x: x.fitness, reverse=True)

	# From Stanley code
	def adjust_speciate_threshold(self):
		re_speciate = False
		
		if len(self.species) < self.params.min_species:
			if self.speciation_adjust_start == False:
				self.speciation_adjust_start = True
				self.distance_threshold_var = -max(0.1, self.params.distance_threshold * 0.01)
			else:
				if self.distance_threshold_var < 0.0:
					self.distance_threshold_var *= 1.05
				else:
					self.distance_threshold_var *= -0.5

			self.params.distance_threshold += self.distance_threshold_var
			self.params.distance_threshold = max(0.1, self.params.distance_threshold)

			re_speciate = True

		elif len(self.species) > self.params.max_species:
			if self.speciation_adjust_start == False:
				self.speciation_adjust_start = True
				self.distance_threshold_var = max(0.1, self.params.distance_threshold * 0.01)
			else:
				if self.distance_threshold_var < 0.0:
					self.distance_threshold_var *= -0.5
				else:
					self.distance_threshold_var *= 1.05

			self.params.distance_threshold += self.distance_threshold_var

			re_speciate = True
		
		else:
			self.speciation_adjust_start = False

		if re_speciate is True:
			self.species = {}
			self.speciate()

	def crossover(self, org_A, org_B):
		# Iterator used to navigate through connection_list and node_list independently for each organism. TODO: use iter()
		itr_A = 0
		itr_B = 0

		# connection_gene's and node_gene's to be selected as a result of crossover
		conn_A = org_A.connection_list[itr_A]
		conn_B = org_A.connection_list[itr_B]

		node_A = org_A.node_list[itr_A]
		node_B = org_B.node_list[itr_B]
		
		# Output genome of the crossover operation. #TODO: enhance genome copy
		new_organism = Genome(num_layers = org_A.num_layers, phenotype = org_A.phenotype)
		new_organism.parent_species = org_A.parent_species

		# Start iterate over genomes to explore the connection_list
		while True:
			if conn_A.innovation > conn_B.innovation:
				new_organism.add_connection(conn_B)
				itr_B += 1
			elif conn_A.innovation == conn_B.innovation:
				if conn_A.enable == True and conn_B.enable == True:
					new_organism.add_connection(random.choice([conn_A, conn_B]))
				elif conn_A.enable == True:
					new_organism.add_connection(conn_A)
				else:
					new_organism.add_connection(conn_B)

				itr_A += 1
				itr_B += 1
			else:
				new_organism.add_connection(conn_A)
				itr_A += 1

			# If iterator is bigger than any of the genome, it means excess
			if itr_A >= len(org_A.connection_list):
				while itr_B < len(org_B.connection_list):
					new_organism.add_connection(org_B.connection_list[itr_B])
					itr_B += 1
				break

			if itr_B >= len(org_B.connection_list):
				while itr_A < len(org_A.connection_list):
					new_organism.add_connection(org_A.connection_list[itr_A])
					itr_A += 1
				break

			conn_A = org_A.connection_list[itr_A]
			conn_B = org_B.connection_list[itr_B]

		itr_A = 0
		itr_B = 0
		
		# Now iterate over node_list
		while True:
			if node_A.gene_id > node_B.gene_id:
				new_organism.add_node(node_B)
				itr_B += 1
			elif node_A.gene_id == node_B.gene_id:
				new_organism.add_node(random.choice([node_A, node_B]))
				itr_A += 1
				itr_B += 1
			else:
				new_organism.add_node(node_A)
				itr_A += 1

			# If iterator is bigger than any of the genome, it means excess
			if itr_A >= len(org_A.node_list):
				while itr_B < len(org_B.node_list):
					new_organism.add_node(org_B.node_list[itr_B])
					itr_B += 1
				break

			if itr_B >= len(org_B.node_list):
				while itr_A < len(org_A.node_list):
					new_organism.add_node(org_A.node_list[itr_A])
					itr_A += 1
				break

			node_A = org_A.node_list[itr_A]
			node_B = org_B.node_list[itr_B]

		return new_organism
	

	def mutate_add_node(self, organism):
		if len(organism.connection_list) == 0:
			return
		
		# Select connection to replace
		connection_replace = random.choice(organism.connection_list)

		if connection_replace.innovation < 0 or connection_replace.enable == False:
			return
		
		# Check if innovation already mutated
		if connection_replace.innovation in self.node_history:
			new_node = self.node_history[connection_replace.innovation].node_connected
			new_node.randomize_function(self.activation_set)
			
			# Load incoming and outgoing innovation from the node_history dict
			incoming_connection_id = self.node_history[connection_replace.innovation].incoming_connection_id
			outgoing_connection_id = self.node_history[connection_replace.innovation].outgoing_connection_id

			# Create new connections
			incoming_connection = ConnectionGene(incoming_connection_id, connection_replace.incoming, new_node.gene_id, 1.0, True, connection_replace.source_layer, new_node.layer)
			outgoing_connection = ConnectionGene(outgoing_connection_id, new_node.gene_id, connection_replace.outgoing, 0.0, True, new_node.layer, connection_replace.target_layer)
			outgoing_connection.randomize_weight()
		else:
			new_node_layer = math.floor(abs(connection_replace.target_layer - connection_replace.source_layer) / 2) + min(connection_replace.source_layer, connection_replace.target_layer)
			if connection_replace.target_layer < new_node_layer or new_node_layer == connection_replace.source_layer or new_node_layer == connection_replace.target_layer:
				return

			new_node = NodeGene(self.get_new_node_id(), NodeType.HIDDEN, self.activation_set.get_random_function(), new_node_layer)

			# Create new connections and increment innovation
			incoming_connection = ConnectionGene(self.get_new_innovation(), connection_replace.incoming, new_node.gene_id, 1.0, True, connection_replace.source_layer, new_node_layer)
			outgoing_connection = ConnectionGene(self.get_new_innovation(), new_node.gene_id, connection_replace.outgoing, 0.0, True, new_node_layer, connection_replace.target_layer)
			outgoing_connection.randomize_weight()

			# Register new additions to node_history
			self.node_history[connection_replace.innovation] = NodeHistoryStruct(incoming_connection.innovation, outgoing_connection.innovation, new_node)
		
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

	def mutate_connection_weight(self, organism):

		for conn in organism.connection_list:
			if random.uniform(0, 1) < self.prob.mutation_weight:
				conn.randomize_weight()
	
	def mutate_node_functions(self, organism):

		for node in organism.node_list:
			if random.uniform(0, 1) < self.prob.mutate_activation and node.node_type != NodeType.INPUT:
				node.randomize_function(self.activation_set)	

	def mutation(self, org): 

		org.recompute_complexity()
		if org.complexity > self.params.small_genome_coeff:
			probabilities_set = {self.prob.lp_new_node: self.mutate_add_node(org),
								self.prob.lp_new_connection: self.mutate_add_connection(org),
								self.prob.mutate_activation: self.mutate_node_functions(org),
								self.prob.mutation_weight: self.mutate_connection_weight(org)}
		else:
			probabilities_set = {self.prob.sp_new_node: self.mutate_add_node(org),
								self.prob.sp_new_connection: self.mutate_add_connection(org),
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

	def reproduce(self):

		self.offspring_organisms = []
		
		for sp in self.species.values():
			if sp.extinct == True:
				continue

			offspring_amount = math.floor(sp.offspring)

			if offspring_amount == 0:
				continue
			
			elite_offspring = round(len(sp.organisms) * self.params.elite_offspring_param)
			elite_count = 0

			if elite_offspring == 0:
				elite_offspring = 1

			while offspring_amount > 0:
				if elite_count < elite_offspring:
					son = copy.deepcopy(sp.best_organism)
					elite_count += 1
				else:
					if random.uniform(0, 1) < self.params.no_crossover_offspring:
						son = copy.deepcopy(self.get_random_organism(sp))
					else:
						random_mother = self.get_random_organism(sp)
						if len(self.species) > 1 and random.uniform(0, 1) < self.prob.interspecies_mating:
							while True:
								random_father = random.choice(list(self.organisms))
								if(sp.birth != random_father.parent_species):
									break
							son = self.crossover(random_mother, random_father)
						else:
							if len(sp.organisms) == 1:
								son = copy.deepcopy(random_mother)
							else:
								while True:
									random_father = self.get_random_organism(sp)
									if(random_mother is not random_father or self.params.allow_clones):
										break
								son = self.crossover(random_mother, random_father)

					self.mutation(son)

				offspring_amount -= 1	
				self.offspring_organisms.append(son)
				
		# Make sure the population is the correct size
		while len(self.offspring_organisms) < self.params.population_max:
			random_species = random.choice(list(self.species.values()))

			random_org = copy.deepcopy(random_species.best_organism)
			random_org.randomize_weights()

			self.offspring_organisms.append(random_org)
		
		self.organisms = self.offspring_organisms

	def speciate(self):

		for sp in self.species.values():
			sp.organisms = []

		for org in self.organisms:
			compatible_species = False

			if len(self.species) == 0:
				new_species = Species()
				new_species.birth = self.get_new_species_id()
				new_species.organisms.append(org)
				
				new_species.best_organism = org
				org.parent_species = new_species.birth # TODO: a method to assign and post-update new count

				self.species[new_species.birth] = new_species
			else:
				if org.parent_species != -1 and org.parent_species in self.species:
					compatibility = self.compatibility(org, self.species[org.parent_species].best_organism)
					if compatibility < self.params.distance_threshold:
						self.species[org.parent_species].organisms.append(org)
						continue

				for sp in self.species.values():
					compatibility = self.compatibility(org, sp.best_organism)
					if compatibility < self.params.distance_threshold:
						compatible_species = True
						org.parent_species = sp.best_organism.parent_species
						sp.organisms.append(org)
						break
				
				if compatible_species is False:
					new_species = Species()
					new_species.birth = self.get_new_species_id()
					new_species.organisms.append(org)
					
					new_species.best_organism = org
					org.parent_species = new_species.birth # TODO: a method to assign and post-update new count

					self.species[new_species.birth] = new_species

	def check_loops(self, organism, inspect, itr):

		for conn in organism.connection_list:
			if conn.incoming == itr:
				if conn.outgoing == inspect:
					return True
				else:
					self.check_loops(organism, inspect, conn.outgoing)
		
		return False

	#def sort_species_by_fitness(self):
	#	self.species.sort(key=lambda x: x.best_fitness, reverse=True)

	def compute_offspring(self):

		pop_shared_fitness = 0.0
		pop_avg_shared_fitness = 0.0

		for sp in self.species.values():
			sp.update_champion()

			sp.age += 1
			sp.offspring = 0.0
			sp.avg_fitness = 0.0

			if sp.best_organism.fitness >= self.champion_fitness:
				self.champion_fitness = sp.best_organism.fitness
				self.champion_genome = copy.deepcopy(sp.best_organism)

			for org in sp.organisms:
				org.shared_fitness = org.fitness / len(sp.organisms)
				sp.avg_fitness += org.shared_fitness		
			
			pop_shared_fitness += sp.avg_fitness

		pop_avg_shared_fitness = pop_shared_fitness / self.params.population_max

		for sp in self.species.values():
			if sp.extinct == True:
				continue

			for org in sp.organisms:
				sp.offspring += org.shared_fitness / pop_avg_shared_fitness

	def remove_empty_species(self):

		species_to_remove = []
		for sp_id in self.species:
			if len(self.species[sp_id].organisms) == 0:
				species_to_remove.append(sp_id)
				continue

		for sp in species_to_remove:
			self.species.pop(sp)

	def get_random_organism(self, species):
		
		if species.extinct == True or len(species.organisms) == 0:
			return None

		survival_cap = math.floor(self.params.survival_threshold * len(species.organisms))

		if survival_cap == 0:
			return species.organisms[0]
		else:
			return random.choice(species.organisms[:survival_cap])

