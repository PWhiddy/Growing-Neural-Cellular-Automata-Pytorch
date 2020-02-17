import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
import random
random.seed(2574)

def to_alpha(x):
    return torch.clamp(x[3:4,:,:], 0.0, 1.0)

def to_rgb(x):
    rgb, a = x[:3,:,:], to_alpha(x)
    return torch.clamp(1.0-a+rgb, 0.0, 1.0)

def show_tensor(t):
    plt.imshow(to_rgb(t).cpu().detach().permute(1,2,0))
    
class CAModel(nn.Module):
    
    def __init__(self, env_d):
        super(CAModel, self).__init__()
        self.conv1 = nn.Conv2d(env_d*3,128,1)
        self.conv2 = nn.Conv2d(128,env_d,1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)
    
class CASimulator():
    
    def __init__(self):
        self.ENV_X = 64
        self.ENV_Y = 64
        self.ENV_D = 16
        self.step_size = 1.0
        self.update_probability = 0.5
        self.cur_batch_size = 8
        self.train_steps = 8000
        self.device = torch.device('cuda')
        self.initial_state = torch.zeros(self.ENV_D, self.ENV_X, self.ENV_Y)
        self.initial_state[3:, self.ENV_X//2, self.ENV_Y//2] = 1.0
        self.initial_state = self.initial_state.to(self.device)
        self.current_states = self.initial_state.repeat(self.cur_batch_size,1,1,1)
        self.current_states = self.current_states.to(self.device)
        self.ca_model = CAModel(self.ENV_D)
        self.ca_model = self.ca_model.to(self.device)
        targ_img = skimage.img_as_float(io.imread('img/snake.png'))
        self.target_states = torch.tensor(targ_img).float().permute(2,0,1).repeat(self.cur_batch_size,1,1,1)
        self.target_states = self.target_states.to(self.device)
        self.optimizer = optim.Adam(self.ca_model.parameters(), lr=2e-3)#lr=2e-3)
        self.frames_out_count = 0
        
    def wrap_edges(self, x):
        return F.pad(x, (1,1,1,1), 'circular', 0)
    
    def living_mask(self):
        alpha = self.current_states[:,3:4,:,:]
        return F.max_pool2d(self.wrap_edges(alpha), 3, stride=1) > 0.1
        
    def raw_senses(self):
        # state - (batch, depth, x, y)
        sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])/8
        sobel_y = torch.tensor([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])/8
        identity = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
        all_filters = torch.stack((identity, sobel_x, sobel_y))
        all_filters_batch = all_filters.repeat(self.ENV_D,1,1).unsqueeze(1)
        all_filters_batch = all_filters_batch.to(self.device)
        return F.conv2d(
            self.wrap_edges(self.current_states), 
            all_filters_batch, 
            groups=self.ENV_D
        )
    
    def sim_step(self):
        pre_update_life_mask = self.living_mask()
        state_updates = self.ca_model(self.raw_senses().to(self.device))*self.step_size
        # randomly block updates to enforce
        # asynchronous communication between cells
        rand_mask = torch.rand_like(
            self.current_states[:, :1, :, :]) < self.update_probability
        self.current_states += state_updates*(rand_mask.float().to(self.device))
        post_update_life_mask = self.living_mask()
        life_mask = pre_update_life_mask & post_update_life_mask
        life_mask = life_mask.to(self.device)
        self.current_states *= life_mask.float()
    
    def run_sim(self, steps, run_idx, save_all):
        self.optimizer.zero_grad()
        for i in range(steps):
            if (save_all):
                show_tensor(self.current_states[0])
                plt.savefig(f'output/all_figs/out{self.frames_out_count:06d}.png')
                self.frames_out_count += 1
            self.sim_step()
        loss = F.mse_loss(self.current_states[:,:4,:,:], self.target_states)
        loss.backward()
        with torch.no_grad():
            self.ca_model.conv1.weight.grad = self.ca_model.conv1.weight.grad/(self.ca_model.conv1.weight.grad.norm()+1e-8)
            self.ca_model.conv1.bias.grad = self.ca_model.conv1.bias.grad/(self.ca_model.conv1.bias.grad.norm()+1e-8)
            self.ca_model.conv2.weight.grad = self.ca_model.conv2.weight.grad/(self.ca_model.conv2.weight.grad.norm()+1e-8)
            self.ca_model.conv2.bias.grad = self.ca_model.conv2.bias.grad/(self.ca_model.conv2.bias.grad.norm()+1e-8)
        self.optimizer.step()
        print(f'loss run {run_idx} : {loss.item()}')
        show_tensor(self.current_states[0])
        plt.savefig(f'output/out{run_idx:03d}.png')
        
    def train_ca(self):
        for idx in range(self.train_steps):
            if (idx < 2000):
                for g in self.optimizer.param_groups:
                    g['lr'] = 2e-3
            elif (idx < 3000):
                for g in self.optimizer.param_groups:
                    g['lr'] = 1e-3
            else:
                for g in self.optimizer.param_groups:
                    g['lr'] = 2e-4
            self.current_states = self.initial_state.repeat(self.cur_batch_size,1,1,1)
            self.run_sim(random.randint(64,96), idx, idx%200 == 0)
            
if __name__ == '__main__':
    ca_sim = CASimulator()
    ca_sim.train_ca()
        