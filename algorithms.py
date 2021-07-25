import ConfigSpace.util as CSU 
import random

class GA:
	def __init__(self,pop,elite):
		self.elite_percentage = elite
		self.population_size = pop
		self.elite_size = int(pop*elite)
		self.best_scores = []
		self.best_configs = []



	def update_elite(self,configs,scores):

		for ID in scores:
			if len(self.best_scores) < self.elite_size:
				self.best_scores.append(scores[ID])
				self.best_configs.append(configs[ID])
			elif scores[ID] > min(self.best_scores):
				idx = self.best_scores.index(min(self.best_scores))
				self.best_scores[idx] = scores[ID]
				self.best_configs[idx] = configs[ID]


	def mutate(self,configs,scores):

		self.update_elite(configs,scores)
		children = []

		#Generate configurations near the best models until the population space is full
		print("elite size: ",len(self.best_configs))
		print(self.elite_size)
		while len(children) < self.population_size:
			children.append(CSU.get_random_neighbor(self.best_configs[random.randint(0,self.elite_size-1)],random.randint(0,9999)))

		return children

		


