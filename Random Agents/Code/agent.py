
import numpy as np
import copy
from scipy.spatial.distance import pdist, squareform, cdist

class Agent:
	''' Create an agent '''

	def __init__(self, position, velocity, radius, mass):	

		self.position = np.array(position)
		self.velocity = np.array([velocity.real, velocity.imag])
		self.radius = radius
		self.mass = mass
		self.memory = [copy.copy(position)]
		self.signal = False
		self.all_bids = {}
		self.signal_position = 0
		self.work = False


	def update_position(self, velocity, min_max):

		if self.work == False:

			if self.signal == False:
				self.velocity = velocity/np.sqrt(velocity[0]**2 + velocity[1]**2)
			if self.signal == True:
				if self.count == 0:
					self.velocity = velocity/np.sqrt(velocity[0]**2 + velocity[1]**2)
				self.count += 1
				if self.count == self.timer:
					self.signal = False
			self.position =	self.position + self.velocity

		else:
			self.signal = False
			self.position =	self.position

		# Extra security to keep the agent inside of the environment in case if something accidentliy push the agent out
		self.position = np.array([ np.min([ np.max([self.position[0], min_max[0][0]]), min_max[0][1]  ]) , \
							np.min([ np.max([self.position[1], min_max[1][0]]), min_max[1][1]  ])    ])
		self.memory.append(copy.copy(self.position))
		return(self.position)


	def working(self, work_position):
		self.work = True
		self.signal = False
		self.work_position = work_position


	def auctioneer(self, bids):
		if len(bids):
			distances = cdist([self.work_position], bids)
			return([np.argsort(distances)[0], distances[0][np.argsort(distances)[0]]])
		else:
			return([])


	def bidder(self, bids):
		if self.work != True:
			self.all_bids[bids[0]] = bids[1]



	def preference_list(self):
		preference_list_out = []
		if len(self.all_bids):
			for k in self.all_bids:
				preference_list_out.append([k, self.all_bids[k]])
			preference_list_out = np.array(preference_list_out)
			# self.all_bids = {}
			return(preference_list_out[np.argsort(preference_list_out[:,1])][:,0])
		else:
			return([])



	def recived_signal(self, timer):
		self.signal = True
		self.timer = timer
		self.count = 0
		self.all_bids = {}


	def calculate_efficiency(self, steps_per_timeunit, timeframe):

		if len(self.memory) > timeframe*steps_per_timeunit:
			epsilon = 0.0001
			step_distance = squareform(pdist(np.vstack(self.memory[-timeframe*steps_per_timeunit + (steps_per_timeunit-1)::steps_per_timeunit])))
			step = step_distance[np.triu_indices(timeframe, k = 1)]
			if len(np.where(step < (steps_per_timeunit-epsilon))[0]) > 0:
				return(len(np.where(step < (steps_per_timeunit-epsilon))[0])/len(step))
			else:
				return(0)
