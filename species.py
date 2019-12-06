from genome import Genome

class Species:

    def __init__(self):
        self.age = 0
        self.offspring_size = 0
        self.best_fitness = 0.0
        self.champion = False
        self.avg_fitness = 0.0
        self.organisms = []

    def sort_by_fitness(self):
        self.organisms.sort(key=lambda x: x.fitness, reverse=True)

    def update_champion(self):
        self.sort_by_fitness()
        self.best_genome = self.organisms[0]
        self.best_fitness = self.best_genome.fitness
