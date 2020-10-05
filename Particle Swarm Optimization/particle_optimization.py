
import numpy as np
from particle import Particle
from enviro import Enviroment
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import copy




def particle_optimization(dim, dim_min, omega, signal, timesteps, plot):
	# Setting up the space
	env1 = Enviroment(dim, 1)
	env1.set_up_coordinates(dim_min)

	signal = [signal + np.random.normal(0, 0, len(dim))]
	
	# Initializing particles
	# particle_positions = [[10, 10], [12, 8], [11, -10], [-4, 9]]
	# initial_velocities = [[1, 0.75], [2, .3], [-.5, 1], [.5, .25]]
	particle_positions = [np.random.randint(dim_min[0], dim_min[0]+dim[0], len(dim)).astype(np.float64) for b in range(10)]
	initial_velocities = [np.random.uniform(-1, 1, 3) for b in range(10)]
	particles = {"Particle"+str(k) : Particle(particle_positions[k], initial_velocities[k], omega) for k in range(len(particle_positions))}



	# Putting in signal and calculating the particle that has shortest distance, and its position
	signal_idx = env1.find_nearest(signal[-1])
	group_best_position = particle_positions[np.argmin([[particles["Particle"+str(k)].calulate_personal_best(signal[-1]),\
											 particles["Particle"+str(k)].personal_best_distance][1]\
									 		 for k in range(len(particle_positions))])]
	group_best_distance = np.min([particles["Particle"+str(k)].personal_best_distance for k in range(len(particle_positions))])
	for k in range(len(particle_positions)):
		particles["Particle"+str(k)].enviroment_position(env1.find_nearest(particles["Particle"+str(k)].particle_position))

	# different_env = {}
	data = []
	time = 250

	for n in range(time):

		personal_bests_distance = np.zeros(len(particle_positions))
		personal_bests_position = np.zeros((len(particle_positions), len(particle_positions[0])))
		data_temp = []

		for k in range(len(particle_positions)):

			data_temp.append(copy.copy(particles["Particle"+str(k)].history[-1]))
			env1.space[particles["Particle"+str(k)].env_current] -= 1
			particles["Particle"+str(k)].position(group_best_position)
			particles["Particle"+str(k)].calulate_personal_best(signal[-1])
			personal_bests_distance[k] = particles["Particle"+str(k)].personal_best_distance
			personal_bests_position[k] = particles["Particle"+str(k)].personal_best_position
			particles["Particle"+str(k)].enviroment_position(env1.find_nearest(particles["Particle"+str(k)].particle_position))

		# Saving timesteps of enviroment, so we can animate it
		# different_env[n] = copy.copy(env1.space)
		data.append(np.array(data_temp))

		# signal.append(signal[-1] + np.random.normal(0, 1, len(dim)))
		signal.append(signal[-1])

		if group_best_distance > np.min(personal_bests_distance):

			group_best_position = particles["Particle"+str(np.argmin(personal_bests_distance))].particle_position
			group_best_distance = np.min(personal_bests_distance)

	data = np.array(data)

	# PLOT
	save = False
	if plot == True:
		# #Animate 2-dim world

		if len(particle_positions[0]) == 2:

			def animate_2dscatters(iteration, data, scatters):
				for i in range(len(data)+1):
					if i < len(data):
						scatters[i].set_offsets( np.c_[[data[i][iteration][0]], [data[i][iteration][1]]] )
					else:
						scatters[i].set_offsets( np.c_[[signal[iteration][0]], [signal[iteration][1]]] )
				return scatters

			fig, ax = plt.subplots() 

			# Initialize scatters
			scatters = [ ax.scatter(particles["Particle"+str(k)].history[0][0], particles["Particle"+str(k)].history[0][1], \
									color='r', s=35) for k in range(len(particle_positions)) ]

			# Insert signal
			scatters.append(ax.scatter(signal[0][0],signal[0][1], color='b', s=35))

			# Number of iterations
			iterations = len(data)

			# Setting the axes properties
			ax.set_xlim([dim_min[0]-(dim[0]*2), dim_min[0]+(dim[0]*3)]); ax.set_xlabel('X')
			ax.set_ylim([dim_min[1]-(dim[1]*2), dim_min[1]+(dim[1]*3)]); ax.set_ylabel('Y')
			ax.set_title('2D Particle Swarm Optimization')
			ax.set_xticks([]); ax.set_yticks([])
			ani = animation.FuncAnimation(fig, animate_2dscatters, iterations, fargs=([particles["Particle"+str(k)].history\
										 for k in range(len(particle_positions))], scatters),
			                             interval=50, blit=False, repeat=True)

			plt.show()
			ani.save('2d_oppgave.gif', dpi=80, writer='imagemagick')

			if save == True:
				Writer = animation.writers['ffmpeg']
				writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
				ani.save('agent2.mp4', writer=writer)



		if len(particle_positions[0]) == 3:

			def animate_3dscatters(iteration, data, scatters):
				for i in range(len(data)+1):
					if i < len(data):
						scatters[i]._offsets3d = ([data[i][iteration][0]], [data[i][iteration][1]], [data[i][iteration][2]])
					else:
						scatters[i]._offsets3d = ([signal[iteration][0]], [signal[iteration][1]], [signal[iteration][2]])
				return scatters

			fig = plt.figure()
			ax = p3.Axes3D(fig)

			# Initialize scatters
			scatters = [ ax.scatter(particles["Particle"+str(k)].history[0][0], particles["Particle"+str(k)].history[0][1], \
									particles["Particle"+str(k)].history[0][2], color='r', s=35) for k in range(len(particle_positions)) ]

			# Insert signal
			scatters.append(ax.scatter(signal[0][0],signal[0][1],signal[0][2], color='b', s=35))

			# Number of iterations
			iterations = len(data)

			# Setting the axes properties
			# ax.set_xlim3d([dim_min[0]-(dim[0]), dim_min[0]+(dim[0]*2)]); ax.set_xlabel('X'); ax.set_ylim3d([dim_min[1]-(dim[1]), dim_min[1]+(dim[1]*2)]); ax.set_ylabel('Y'); ax.set_zlim3d([dim_min[2]-(dim[2]), dim_min[2]+(dim[2]*2)]); ax.set_zlabel('Z')
			ax.set_xlim3d([dim_min[0]-(dim[0]/2), dim_min[0]+(dim[0]*1.5)]); ax.set_xlabel('X'); ax.set_ylim3d([dim_min[1]-(dim[1]/2), dim_min[1]+(dim[1]*1.5)]); ax.set_ylabel('Y'); ax.set_zlim3d([dim_min[2]-(dim[2]/2), dim_min[2]+(dim[2]*1.5)]); ax.set_zlabel('Z')
			ax.set_title('3D Particle Swarm Optimization'); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
			ani = animation.FuncAnimation(fig, animate_3dscatters, iterations, fargs=([particles["Particle"+str(k)].history for k in range(len(particle_positions))], scatters),
			                                   interval=50, blit=False, repeat=True)
			plt.show()


			# ani.save('3d.gif', dpi=80, writer='imagemagick')

			if save == True:
				Writer = animation.writers['ffmpeg']
				writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
				ani.save('agent2.mp4', writer=writer)




dim = [30, 30, 30]
dim_min = [-15, -15, -15]
timesteps = 250
plot = True
omega = [0.98, 0.04, 0.081]
signal = [10, 0, -5]
particle_optimization(dim, dim_min, omega, signal, timesteps, plot)


