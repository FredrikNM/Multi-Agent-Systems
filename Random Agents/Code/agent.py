
import numpy as np
import copy
from scipy.spatial.distance import pdist, squareform

class Agent:
	''' Create an agent '''

	def __init__(self, position, velocity, radius, mass):	

		self.position = np.array(position)
		self.velocity = np.array([velocity.real, velocity.imag])
		self.radius = radius
		self.mass = mass
		self.memory = [copy.copy(position)]
		self.signal = False
		self.work = False


	def update_position(self, velocity, min_max):

		if self.work == False:

			if self.signal == False:
				self.velocity = velocity
			if self.signal == True:
				if self.count == 0:
					self.velocity = velocity
				self.count += 1
				if self.count == self.timer:
					self.signal = False
		else:
			self.velocity = np.array([0, 0])

		self.position =	self.position + self.velocity
		# Extra security to keep the agent inside of the environment if something accidentliy push the agent out
		self.position = np.array([ np.min([ np.max([self.position[0], min_max[0][0]]), min_max[0][1]  ]) , \
							np.min([ np.max([self.position[1], min_max[1][0]]), min_max[1][1]  ])    ])
		self.memory.append(copy.copy(self.position))
		return(self.position)


	def working(self):
		self.work = True
		self.signal = False


	def recived_signal(self, timer):
		self.signal = True
		self.timer = timer
		self.count = 0


	def calculate_efficiency(self, steps_per_timeunit, timeframe):

		if len(self.memory) > timeframe*steps_per_timeunit:
			epsilon = 0.0001
			step_distance = squareform(pdist(np.vstack(self.memory[-timeframe*steps_per_timeunit + (steps_per_timeunit-1)::steps_per_timeunit])))
			step = step_distance[np.triu_indices(timeframe, k = 1)]
			if len(np.where(step < (steps_per_timeunit-epsilon))[0]) > 0:
				return(len(np.where(step < (steps_per_timeunit-epsilon))[0])/len(step))
			else:
				return(0)
