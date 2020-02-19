from genome import Genome

class Species:

    def __init__(self, copy=None):
        self.age = 0
        self.offspring = 0.0
        self.best_fitness = 0.0
        self.champion = False
        self.avg_fitness = 0.0
        self.organisms = []
        self.birth = 0

    def sort_by_fitness(self):
        self.organisms.sort(key=lambda x: x.fitness, reverse=True)

    def update_champion(self):
        self.sort_by_fitness()
        self.best_genome = self.organisms[0]
        self.best_fitness = self.best_genome.fitness

    def __eq__(self, other): 
        if(self.birth == other.birth): 
            return True
        else: 
            return False
