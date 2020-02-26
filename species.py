from genome import Genome
import copy

class Species:

    def __init__(self, copy=None):
        self.age = 0
        self.offspring = 0.0
        self.birth = 0
        self.extinct = False

        self.avg_fitness = 0.0
        self.best_fitness = 0.0
        
        self.organisms = []
        self.best_organism = None

        self.champion = False

    def sort_by_fitness(self):
        self.organisms.sort(key=lambda x: x.fitness, reverse=True)

    def update_champion(self):
        self.sort_by_fitness()
        self.best_organism = copy.deepcopy(self.organisms[0])
        self.best_fitness = self.best_organism.fitness

    def __eq__(self, other): 
        if(self.birth == other.birth): 
            return True
        else: 
            return False
