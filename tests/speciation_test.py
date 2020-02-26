from py-hyperneat.population import Population

params = open('testConfig.json', 'r')
genome = open('Champion.json', 'r')

pop = Population()
pop.start_generation(genome, params)

print(pop.current_species)


