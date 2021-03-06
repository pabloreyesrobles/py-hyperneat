from neat.species import Species
from neat.genome import Genome
from neat.genes import ConnectionGene, NodeGene, NodeType
from neat.activation_functions import ActivationFunction
from neat.neural_network import Neuron, Connection, NeuralNetwork
from neat.population import Population, NodeHistoryStruct

import json
import copy
import random
import math
import numpy as np
from dataclasses import dataclass

class TrainTask:
	PREDICTION = 0,
	CLASSIFICATION = 1

class Neat:
	
	def __init__(self, fitness_eval=None, train_task=None, max_generation=0):		
		self.pop = Population()

		self.current_generation = 0
		self.max_generation = max_generation
		self.best_historical_fitness = 0.0
		
		self.fitness_eval = fitness_eval
		self.train_task = train_task

		self.input_data = None
		
		#TODO: to be replaced
		self.avg_fitness = 0.0
		self.best_epoch_fitness = 0.0
		# Must be set True to operate
		self.configurated = False

		self.historical_avg_fitness = []
		self.historical_best_fitness = []

	def import_config(self, config_file, genome_file):		
		try:
			config = json.load(config_file)
		except ValueError:
			print('NEAT: Invalid config file')
			return False

		genome = Genome()
		if genome.import_genome(genome_file) is False:
			raise NameError('Cant load base genome')

		self.max_generation = config['generations']
		self.pop.config_population(config)
		self.pop.start_population(genome)

		self.configurated = True

	def set_multi_input(self, data):		
		if self.train_task == TrainTask.PREDICTION:
			if len(data) <= 1:
				raise Exception('Use set_single_input')

			self.input_data = data

	def evaluate_population(self, net=None):	
		self.avg_fitness = 0.0
		self.best_epoch_fitness = 0.0

		for org in self.pop.organisms:
			net = org.build_phenotype()

			org.fitness = self.fitness_eval(self.input_data, net)
			self.avg_fitness += org.fitness 
			if org.fitness > self.best_epoch_fitness:
				self.best_epoch_fitness = org.fitness

		self.avg_fitness /= self.pop.params.population_max

	# TODO:
	# Best_fitness, historical_fitness, age mechanism needed
	# champion_species tracking
	# stagnation respect to best historical fitness
	def epoch(self, print_stats=False):
		self.pop.sort_organisms()
		self.historical_avg_fitness.append(self.avg_fitness)
		self.historical_best_fitness.append(self.best_epoch_fitness)

		self.pop.adjust_speciate_threshold()
		self.pop.compute_offspring()

		self.pop.reproduce()
		self.pop.speciate()

		self.pop.remove_empty_species()

		for sp in self.pop.species.values():
			print('Species #{} size: {} avg_fitness: {:.4f} best_fitness: {:.4f} extinct: {}'.format(sp.birth, len(sp.organisms), sp.avg_fitness, sp.best_fitness, sp.extinct))
		
		self.epoch_info = 'Generation #{:d}: species = {:d}, champion_fitness = {:f}, avg_generation_fitness = {:f}\n'.format(self.current_generation, len(self.pop.species), self.pop.champion_fitness, self.avg_fitness)
		if print_stats == True:
			print(self.epoch_info)

		self.current_generation += 1

	def run(self):
		while self.current_generation < self.max_generation:
			self.evaluate_population()    
			self.epoch()

		return self.historical_avg_fitness, self.historical_best_fitness

	def run_multiple_trainings(self, num_trainings):
		multiple_hist_avg_fitness = np.zeros(self.max_generation)
		multiple_hist_best_fitness = np.zeros(self.max_generation)

		current_training = 0

		while current_training < num_trainings:
			self.run()
			print('Iteration #{:d}: species = {:d}, champion_fitness = {:f}, avg_generation_fitness = {:f}'.format(current_training, len(self.pop.species), self.pop.champion_fitness, self.avg_fitness))

			multiple_hist_avg_fitness += np.array(self.historical_avg_fitness)
			multiple_hist_best_fitness += np.array(self.historical_best_fitness)

			self.pop.restart_population()

			self.historical_avg_fitness = []
			self.historical_best_fitness = []
			self.current_generation = 0

			current_training += 1

		return multiple_hist_avg_fitness / num_trainings, multiple_hist_best_fitness / num_trainings
		