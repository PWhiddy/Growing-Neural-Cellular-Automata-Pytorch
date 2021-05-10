import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
import random
import math
import argparse
from torchvision import transforms
from Particle_Sim import ParticleSystem
random.seed(2574)

def to_rgb(x):
    rgb = x[0:3,:,:]
    return torch.clamp(rgb, 0.0, 1.0)

def show_tensor_surfaces(t):
    if (len(t.shape) < 4):
        plt.axis('off')
        plt.set_cmap('inferno')
        plt.imshow(to_rgb(t).cpu().detach().permute(1,2,0), interpolation='nearest')
    else:
        # batch must be multiple of 2
        plt.set_cmap('inferno')
        fig, axs = plt.subplots(4,t.shape[0]//4, figsize=(8, 8))
        plt.subplots_adjust(hspace =0.02, wspace=0.02)
        for axe,batch_item in zip(axs.flatten(),t):
            axe.axis('off')
            axe.imshow(to_rgb(batch_item).cpu().detach().permute(1,2,0), interpolation='nearest')

def state_loss(running_state, final_state):
    return F.mse_loss(running_state[:,0:3,:,:], final_state[:,0:3,:,:])

class CAModel(nn.Module):
    
    def __init__(self, env_d):
        super(CAModel, self).__init__()
        self.conv1 = nn.Conv2d(env_d*3,96,1)
        self.conv2 = nn.Conv2d(96,env_d,1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)
    
class CASimulator():
    
    def __init__(self):
        self.ENV_X = 32
        self.ENV_Y = 32
        self.ENV_D = 12
        self.particle_count = 9
        self.init_particle_dim = 3
        self.particle_size = 5.0
        self.part_spacing = 1.4
        self.repel_strength = 0.012
        self.step_size = 1.0
        self.update_probability = 0.5
        self.cur_batch_size = 16
        self.train_steps = 64000
        self.min_sim_steps = 6
        self.max_sim_steps = 6
        self.step_increase_interval = 128
        self.updates_per_step = 8
        self.comp_loss_interval = 8
        self.device = torch.device('cuda')
        self.ca_model = CAModel(self.ENV_D)
        self.ca_model = self.ca_model.to(self.device)

        self.optimizer = optim.Adam(self.ca_model.parameters(), lr=2e-3)
        self.frames_out_count = 0
        self.losses = []
        self.writer = SummaryWriter(log_dir='LR/runs/parts_v2_d')
        self.checkpoint_interval = 500
        self.final_plot_interval = 1
        self.evolution_interval = 128
        # lr decay + sigmoid warm-up for training restarts
        self.lr_schedule = lambda x: (1/(1+math.exp(-0.2*x+5)))*1e-3*2.0**(-0.0002*x) #lambda x: 2e-3 if x<4000 else 3e-4

    def initialize_particle_sims(self):
        self.p_sims = [
            ParticleSystem(
                random.randint(3,self.particle_count), 
                self.ENV_X, 
                self.particle_size, 
                self.repel_strength,
                self.init_particle_dim,
                self.part_spacing,
                i
            ) 
            for i in range(self.cur_batch_size)
        ]

    def draw_states(self):
        blank = torch.zeros(self.cur_batch_size, self.ENV_D, self.ENV_X, self.ENV_Y, device=self.device)
        blank[:,0:3,:,:] = torch.tensor([ps.draw() for ps in self.p_sims], device=self.device).permute(0,3,1,2)
        return blank

    def run_particles(self, num_steps):
        for ps in self.p_sims:
            for _ in range(num_steps):
                ps.sim()

    def load_pretrained(self, path):
        self.ca_model.load_state_dict(torch.load(path))

    def wrap_edges(self, x):
        return F.pad(x, (1,1,1,1), 'constant', 0.0) #'circular', 0)

    def raw_senses(self):
        # state - (batch, depth, x, y)
        sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]], device=self.device)/8
        sobel_y = torch.tensor([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]], device=self.device)/8
        identity = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]], device=self.device)
        all_filters = torch.stack((identity, sobel_x, sobel_y))
        all_filters_batch = all_filters.repeat(self.ENV_D,1,1).unsqueeze(1)
        all_filters_batch = all_filters_batch
        return F.conv2d(
            self.wrap_edges(self.current_states), 
            all_filters_batch, 
            groups=self.ENV_D
        )
    
    def sim_step(self):
        state_updates = self.ca_model(self.raw_senses())*self.step_size
        # randomly block updates to enforce
        # asynchronous communication between cells
        rand_mask = torch.rand_like(
            self.current_states[:, :1, :, :], device=self.device) < self.update_probability
        self.current_states += state_updates*(rand_mask.float())

    def run_sim(self, steps, run_idx, save_all):
        self.optimizer.zero_grad()
        for i in range(steps):
            if (save_all):
                show_tensor_surfaces(self.current_states)
                plt.savefig(f'output/all_figs/out_hidden_{self.frames_out_count:06d}.png')
                plt.close('all')
                self.frames_out_count += 1
            self.sim_step()
            self.run_particles(1)
        
            if ((i+1) % self.comp_loss_interval == 0):
                self.final_states = self.draw_states()
                loss = state_loss(self.current_states, self.final_states)
                loss.backward(retain_graph=(i+1) != steps)
            #self.set_unique_control_channel()

        self.optimizer.step()
        lsv = loss.item()
        self.losses.insert(0, lsv)
        self.losses = self.losses[:100]
        self.writer.add_scalar('Loss/batch', lsv, run_idx)
        self.writer.add_scalar('Loss/moving_average', sum(self.losses)/len(self.losses), run_idx)
        self.writer.add_scalar('LearningRate/val', self.lr_schedule(run_idx), run_idx)
        

    def train_ca(self):
        self.initialize_particle_sims()
        for idx in range(self.train_steps):
            
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr_schedule(idx)
            
            #self.current_states = self.initial_state.repeat(self.cur_batch_size,1,1,1)
            #self.initialize_particle_sims()
            self.current_states = self.draw_states()
            num_steps = self.max_sim_steps*self.updates_per_step #random.randint(self.min_sim_steps,min(idx//self.step_increase_interval+1,self.max_sim_steps))*self.updates_per_step
            self.run_sim(num_steps, idx, (idx+1)%self.evolution_interval == 0)
            if (idx % self.final_plot_interval == 0):
                #show_final_target(self.input_matsA, self.input_matsB, self.current_states)
                show_tensor_surfaces(self.current_states)
                #show_tensor_surfaces(self.final_states)
                plt.savefig(f'output/out{idx:06d}.png')
                plt.close('all')
            if (idx % self.checkpoint_interval == 0):
                torch.save(self.ca_model.state_dict(), f'checkpoints/ca_model_step_{idx:06d}.pt')

    def run_pretrained(self, steps):
        #self.cur_batch_size = 1
        self.initialize_particle_sims()
        with torch.no_grad():
            self.current_states = self.draw_states()
            for idx in range(steps):
                print(f'step: {idx}')
                if (idx % 8 == 0):
                    show_tensor_surfaces(self.current_states[0])
                    plt.savefig(f'pretrained/out_{self.frames_out_count:06d}.png')
                    plt.close('all')
                    self.frames_out_count += 1
                self.sim_step()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run-pretrained', dest='run_pretrained', action='store_true')

    parser.add_argument('--pretrained-path', type=str, default='ca_model_pretty_g_short')

    args = parser.parse_args()

    ca_sim = CASimulator()

    if args.run_pretrained:
        print('running pretained')
        ca_sim.load_pretrained(f'checkpoints/{args.pretrained_path}.pt')
        ca_sim.run_pretrained(50000)
    else:
        #ca_sim.load_pretrained(f'checkpoints/ca_model_short_d.pt')
        ca_sim.train_ca()
        
