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

class TrainTask:
	PREDICTION = 0,
	CLASSIFICATION = 1

class Neat:

	def __init__(self, fitness_eval=None, train_task=None):
		
		self.pop = Population()

		self.current_generation = 0
		self.max_generation = 0
		self.best_historical_fitness = 0.0
		
		self.fitness_eval = fitness_eval
		self.train_task = train_task

		self.input_data = None
		
		#TODO: to be replaced
		self.avg_fitness = 0.0
		self.best_epoch_fitness = 0.0
		# Must be set True to operate
		self.configurated = False

	def import_config(self, config_file, genome_file):
		
		try:
			config = json.load(config_file)
		except ValueError:
			print('Invalid config file')
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

	def evaluate_population(self):
		
		self.avg_fitness = 0.0
		self.best_epoch_fitness = 0.0

		for org in self.pop.organisms:
			net = org.build_layered_phenotype()

			org.fitness = self.fitness_eval(self.input_data, net)
			self.avg_fitness += org.fitness
			if org.fitness > self.best_epoch_fitness:
				self.best_epoch_fitness = org.fitness

	# TODO:
	# Best_fitness, historical_fitness, age mechanism needed
	# champion_species tracking
	# stagnation respect to best historical fitness
	def epoch(self):

		self.pop.adjust_speciate_threshold()
		self.pop.compute_offspring()

		self.pop.reproduce()
		self.pop.speciate()

		self.pop.remove_empty_species()
		
		self.current_generation += 1


		