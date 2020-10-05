

import numpy as np

class intensity_circle_plot:

	# Class to fix how a circle is discretize on a grid (how many gridpoints is the radius equal to)
	# We set the grid so we can have a "round" circle, meaning the grid-radius relation
	# gives us a gridspace that is odd numbered such that the circle will have a center. 
	# This is made for the specific case of a rectangular environment for x between (0, some width),
	# y between (0, some hight). The objects we want a radius around is sent to the intensity function which
	# gives a the intensity as a matrix. If you want to use it for other cases that I have done,
	# you probably must modify the intensity function, since now it loops over agents called A0, A1, ect, 
	# that has the x, y coordinates

	def __init__(self, radius, discretization, walls):	
		self.grid_size = int(radius/discretization)	# Descretiziation of mesh for heatmap
		if self.grid_size < 1:
			self.grid_size = 1	# In case our cirle is discretizate to 0
		self.walls = walls	# Walls of our environment
		self.grid_x_max = walls[0][1]
		self.grid_y_max = walls[1][1]

		def radius_grid(radius, grid_size):

			# Radius grid relation
			return( int(radius*2/grid_size)+1 )

		def e_or_o(number):

			# Even or odd number
			return(number % 2 == 0)


		# If the relation is even we want to fix our grid such that the space have odd number length and height
		while e_or_o(radius_grid(radius, self.grid_size)):

			delta_grid_size = self.grid_size/100
			grid_temp = self.grid_size - delta_grid_size

			# Check that we dont move to much. Remember we only want to add an extra slot to our grid
			while abs(radius_grid(radius, grid_temp) - radius_grid(radius, self.grid_size)) > 1:
				delta_grid_size = delta_grid_size/10
				grid_temp = self.grid_size - delta_grid_size

			self.grid_size = grid_temp

		self.n_points = radius_grid(radius, self.grid_size)
		self.radius_n_points = int(self.n_points/2)

		def circular_mesh(radius, grid_size, n_points):

			''' Circle made out of radius and descretizized  '''
			x_grid = np.linspace(0, (radius*2), int(radius*2/grid_size)+1   )
			base_circle = np.zeros(( len(x_grid), len(x_grid) ))
			[base_circle.__setitem__((i, j), 1) for i in range(len(base_circle)) for j in range(len(base_circle)) \
			if np.sqrt((i-int(len(base_circle)/2))**2 + (j-int(len(base_circle)/2))**2) <= int(len(base_circle)/2)]
			return(base_circle)

		self.base_circle = circular_mesh(radius, self.grid_size, self.n_points)
		antall_x_punkter = int(self.grid_x_max/self.grid_size) + 1; antall_y_punkter = int(self.grid_y_max/self.grid_size) + 1
		x_grid = np.linspace(0, self.grid_x_max, antall_x_punkter); y_grid = np.linspace(0, self.grid_y_max, antall_y_punkter)
		self.x_mesh, self.y_mesh = np.meshgrid(x_grid, y_grid)
		self.width_min = 0 ; self.width_max = len(self.x_mesh)
		self.height_min = 0 ; self.height_max = len(self.y_mesh)
		self.new_zeros = np.zeros((self.width_max, self.height_max))


	def idx_to_circle(self, width_min, width_max, height_min, height_max, x, y):
		# Returns the indexes to the circle in the gridspace

		idx_w = np.arange(self.n_points)
		if x < (width_min + self.radius_n_points):
		 	idx_w = np.arange( abs( self.radius_n_points - x ), self.n_points )

		if x >= (width_max - (self.radius_n_points+1)):
		 	idx_w = np.arange(0 , self.n_points - (x - (width_max - (self.radius_n_points+1)))  )

		idx_h = np.arange(self.n_points)
		if y < (height_min + self.radius_n_points):
		 	idx_h = np.arange( abs( self.radius_n_points - y ), self.n_points )

		if y >= (height_max - (self.radius_n_points+1)):
		 	idx_h = np.arange(0 , self.n_points - (y - (height_max - (self.radius_n_points+1)))  )

		return(idx_w, idx_h)


	def intensity(self, agents):

		for k in range(len(agents)):
			# Modify either your data to fit this structure, or this line to fit your need
			x, y = agents["A"+str(k)].position
			# Ensuring that the x, y coordinates are inside of the grid, else we would get errors
			x, y = np.array([ np.min([ np.max([x, self.walls[0][0]]), self.walls[0][1]  ]) , \
								np.min([ np.max([y, self.walls[1][0]]), self.walls[1][1]  ])    ])

			x = int(x/self.grid_size)
			y = int(y/self.grid_size)

			idx_w, idx_h = self.idx_to_circle(self.width_min, self.width_max, self.height_min, self.height_max, x, y)
			self.new_zeros[np.max([self.width_min, x-self.radius_n_points]):np.min([self.width_max,x+self.radius_n_points+1]), \
					  np.max([self.height_min, y-self.radius_n_points]):np.min([self.height_max, y+self.radius_n_points+1])] += self.base_circle[idx_w,:][:,idx_h]

		return(self.new_zeros.T)

