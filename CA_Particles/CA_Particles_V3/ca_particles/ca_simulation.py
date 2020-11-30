import torch
import torch.nn.functional as F

from ca_particles import Simulation

class CASimulation(Simulation):
    
    def __init__(self, ca_model, device, batch_size, env_size=32, env_depth=16, update_prob=0.5):
        self.model = ca_model
        self.env_size = env_size
        self.env_depth = env_depth
        self.device = device
        self.batch_size = batch_size
        self.update_probability = update_prob
        self.reset()
        
    def reset(self):
        self.states = torch.zeros(
            self.batch_size, self.env_depth, self.env_size, self.env_size,
            device=self.device)
        
    def wrap_edges(self, x):
        return F.pad(x, (1,1,1,1), 'circular', 0)

    def raw_senses(self):
        # states - (batch, depth, x, y)
        sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]], device=self.device)/8
        sobel_y = torch.tensor([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]], device=self.device)/8
        identity = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]], device=self.device)
        all_filters = torch.stack((identity, sobel_x, sobel_y))
        all_filters_batch = all_filters.repeat(self.env_depth,1,1).unsqueeze(1)
        return F.conv2d(
            self.wrap_edges(self.states), 
            all_filters_batch, 
            groups=self.env_depth
        )
        
    def sim_step(self, time_step):
        states_updates = self.model(self.raw_senses())*time_step
        # randomly block updates to enforce
        # asynchronous communication between cells
        rand_mask = torch.rand_like(
            self.states[:, :1, :, :], device=self.device) < self.update_probability
        self.states += states_updates*(rand_mask.float())
             
    def draw(self):           
        return self.states[:,0:3,:,:]