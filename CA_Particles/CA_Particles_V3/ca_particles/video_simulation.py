import numpy as np

from ca_particles import Simulation, Particle, Walls

class VideoSimulation(Simulation):
    '''
    Batched Pre-rendered simulation
    Takes a numpy like array representing a video
    of a grid simulation and steps through them
    '''
    
    def __init__(self, data_arr, batch_size, model_steps_per_frame):
        self.sim_data = data_arr
        print(f'Initialized VideoSimulation with data: {data_arr.shape}')
        self.t_steps = data_arr.shape[0]
        self.grid_height = data_arr.shape[1]
        self.grid_width = data_arr.shape[2]
        self.batch_size = batch_size
        self.model_steps_per_frame = model_steps_per_frame
        self.initial_indices = np.arange(0,batch_size)*(self.t_steps-(self.t_steps%batch_size))//batch_size
        self.reset()
        
    def reset(self):
        self.cur_indices = np.copy(self.initial_indices)
        self.sim_steps = 0
    
    def sim_step(self, time_step):
        if self.sim_steps % self.model_steps_per_frame == 0:
            self.cur_indices += 1
            self.cur_indices = np.mod(self.cur_indices, self.t_steps)
        self.sim_steps += 1
        
    def draw(self):
        return self.sim_data[self.cur_indices]
        