from neat.population import Population
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

params = open('tests/config_files/testConfig.json', 'r')
genome = open('tests/config_files/Champion.json', 'r')

pop = Population()
pop.start_population(genome)

print(pop.current_species)


