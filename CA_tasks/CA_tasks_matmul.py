import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
import random
import math
import argparse
from opensimplex import OpenSimplex
from torchvision import transforms
random.seed(2574)

def to_rgb(x):
    rgb = x[0,:,:]
    return torch.clamp(rgb*0.15, 0.0, 1.0)

def show_tensor(t):
    plt.axis('off')
    plt.set_cmap('inferno')
    plt.imshow(to_rgb(t).cpu().detach(), interpolation='nearest')

def show_tensor_surfaces(t):
    if (len(t.shape) < 4):
        plt.axis('off')
        plt.set_cmap('inferno')
        plt.imshow(to_rgb(0.9*t).cpu().detach(), interpolation='nearest')
    else:
        # batch must be multiple of 2
        plt.set_cmap('inferno')
        fig, axs = plt.subplots(4,t.shape[0]//4, figsize=(8, 16))
        plt.subplots_adjust(hspace =0.02, wspace=0.02)
        for axe,batch_item in zip(axs.flatten(),t):
            axe.axis('off')
            axe.imshow(to_rgb(0.9*batch_item).cpu().detach(), interpolation='nearest')

def show_hidden(t, section):
    if (len(t.shape) < 4):
        plt.axis('off')
        plt.set_cmap('inferno')
        plt.imshow(torch.sigmoid(t[1:4,:,:]).cpu().detach().permute(1,2,0), interpolation='nearest')
    else:
        # batch must be multiple of 2
        plt.set_cmap('inferno')
        fig, axs = plt.subplots(4,t.shape[0]//4, figsize=(16, 16))
        plt.subplots_adjust(hspace =0.01, wspace=0.1)
        for axe,batch_item in zip(axs.flatten(),t):
            axe.axis('off')
            axe.imshow(torch.sigmoid(batch_item[1:4,:,:]).cpu().detach().permute(1,2,0), interpolation='nearest')

def convert_image(t,section):
    tf = transforms.Normalize(mean=[0.0,0.0,0.0], std=[0.33,0.33,0.33])
    return torch.clamp(tf(t[0,section*3+1:section*3+4,:,:]),0.0,1.0).cpu().detach().permute(1,2,0)

def show_all_layers(t,o):
    plt.set_cmap('inferno')
    fig, axs = plt.subplots(8,2, figsize=(8, 32))
    plt.subplots_adjust(hspace =0.02, wspace=0.02)

    axs[0,0].axis('off')
    axs[0,0].imshow(convert_image(t, 0), interpolation='nearest')
    axs[0,1].axis('off')
    axs[0,1].imshow(to_rgb(o[0]).cpu().detach(), interpolation='nearest')
    axs[1,1].axis('off')
    axs[1,0].axis('off')
    if (t.shape[1] > 4):
        axs[1,0].imshow(convert_image(t, 1), interpolation='nearest')    
        axs[1,1].imshow(to_rgb(o[0]).cpu().detach(), interpolation='nearest')
    axs[2,0].axis('off')
    axs[2,1].axis('off')
    if (t.shape[1] > 7):
        axs[2,0].imshow(convert_image(t, 2), interpolation='nearest')
        axs[2,1].imshow(to_rgb(o[0]).cpu().detach(), interpolation='nearest')
    axs[3,0].axis('off')
    axs[3,1].axis('off')
    if (t.shape[1] > 10):
        axs[3,0].imshow(convert_image(t, 3), interpolation='nearest')
        axs[3,1].imshow(to_rgb(o[0]).cpu().detach(), interpolation='nearest')

    axs[4,0].axis('off')
    axs[4,1].axis('off')
    if (t.shape[1] > 13):
        axs[4,0].imshow(convert_image(t, 4), interpolation='nearest')
        axs[4,1].imshow(to_rgb(o[0]).cpu().detach(), interpolation='nearest')
    axs[5,0].axis('off')
    axs[5,1].axis('off')
    if (t.shape[1] > 16):
        axs[5,0].imshow(convert_image(t, 5), interpolation='nearest')
        axs[5,1].imshow(to_rgb(o[0]).cpu().detach(), interpolation='nearest')
    axs[6,0].axis('off')
    axs[6,1].axis('off')
    if (t.shape[1] > 19):
        axs[6,0].imshow(convert_image(t, 6), interpolation='nearest')
        axs[6,1].imshow(to_rgb(o[0]).cpu().detach(), interpolation='nearest')
    axs[7,0].axis('off')
    axs[7,1].axis('off')
    if (t.shape[1] > 21):
        axs[7,0].imshow(convert_image(t, 7), interpolation='nearest')
        axs[7,1].imshow(to_rgb(o[0]).cpu().detach(), interpolation='nearest')

def make_initial_state(batch_size,d,x,y):
    i_state = torch.zeros(batch_size, d,x,y)
    i_state[:, 1:, x//2-2, y//2-2] = 1.0
    return i_state

def make_input_fractal(batch_size, x, y):
    inp = torch.zeros(batch_size,x,y)
    for val in inp:
        # assume pow of two
        smp = OpenSimplex(random.randint(0,1e10))
        for lvl in range(8):
            amp = 0.5**lvl
            freq = 0.04*2**lvl
            for a in range(x):
                for b in range(y):
                    val[a][b] += amp*smp.noise2d(x=freq*a, y=freq*b)
        val *= 10.0
        val.sin_()
        val *= 0.5
        val += 0.5
    return inp 

def make_input(batch_size,x,y):
    inp = 0.5*torch.rand(batch_size,x,y)
    for batch_item in inp:
        batch_item += torch.rand(1)*0.5
    return inp

def make_final_state(input_state_A, input_state_B, running_state):
    final_state = running_state.clone()
    final_state[:,0,
        2:2+input_state_A.shape[1], 
        26:26+input_state_A.shape[2]] = input_state_B
    final_state[:,0,
        26:26+input_state_B.shape[1], 
        2:2+input_state_B.shape[2]] = input_state_A #torch.flip(input_state, [0,1])
    input_mult = torch.matmul(input_state_A, input_state_B)
    final_state[:,0,
        26:26+input_mult.shape[1], 
        26:26+input_mult.shape[2]] = input_mult
    return final_state

def reset_to_input(input_state_A, input_state_B, running_state):
    # set input values
    running_state[:,0,
        2:2+input_state_A.shape[1], 
        26:26+input_state_A.shape[2]] = input_state_B
    running_state[:,0,
        26:26+input_state_B.shape[1], 
        2:2+input_state_B.shape[2]] = input_state_A

def show_final_target(input_state_A, input_state_B, running_state):
    input_mult = torch.matmul(input_state_A, input_state_B)
    running_state[:,0,
        2:2+input_mult.shape[1], 
        2:2+input_mult.shape[2]] = input_mult
    
    '''
    # this breaks it!
    # seems that the initial task of learning box shape
    # actually allows it to then solve the more difficult problem
    # of transfering information!
    # clear input/output channel
    running_state[:,1,:,:] = 0.0
    # designate input area
    running_state[:,1,
        3:3+input_state.shape[1], 
        3:3+input_state.shape[2]] = -1.0
    # designate output area
    running_state[:,1,
        5+input_state.shape[1]:5+2*input_state.shape[1], 
        5+input_state.shape[2]:5+2*input_state.shape[2]] = 1.0
    '''

def state_loss(running_state, final_state):
    return F.mse_loss(running_state[:,0,:,:], final_state[:,0,:,:])
    
class CAModel(nn.Module):
    
    def __init__(self, env_d):
        super(CAModel, self).__init__()
        self.conv1 = nn.Conv2d(env_d*3,232,1)
        self.conv2 = nn.Conv2d(232,env_d,1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)
    
class CASimulator():
    
    def __init__(self):
        self.ENV_X = 50
        self.ENV_Y = 50
        self.ENV_D = 25
        self.input_x = 22
        self.input_y = 22
        self.step_size = 1.0
        self.update_probability = 0.5
        self.cur_batch_size = 8
        self.train_steps = 64000
        self.min_sim_steps = 96#48
        self.max_sim_steps = 128#64
        self.device = torch.device('cuda')
        self.ca_model = CAModel(self.ENV_D)
        self.ca_model = self.ca_model.to(self.device)

        self.optimizer = optim.Adam(self.ca_model.parameters(), lr=2e-3)
        self.frames_out_count = 0
        self.losses = []
        self.checkpoint_interval = 500
        self.final_plot_interval = 10
        self.evolution_interval = 500
        self.lr_schedule = lambda x: 2e-3*2.0**(-0.0002*x) #lambda x: 2e-3 if x<4000 else 3e-4

    def load_pretrained(self, path):
        self.ca_model.load_state_dict(torch.load(path))
        
    def wrap_edges(self, x):
        return F.pad(x, (1,1,1,1), 'constant', 0.0) #'circular', 0)
        
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
        state_updates = self.ca_model(self.raw_senses().to(self.device))*self.step_size
        # randomly block updates to enforce
        # asynchronous communication between cells
        rand_mask = torch.rand_like(
            self.current_states[:, :1, :, :]) < self.update_probability
        self.current_states += state_updates*(rand_mask.float().to(self.device))

    def set_experiment_control_channel(self, s_index):
        # set image target channels
        self.current_states[:,-self.target_count:,:,:] = 0.0
        #self.current_states[:,-3] = 1.0
        # swap x and y here?
        #self.current_states[:,-2] = torch.clamp(torch.linspace(-1,2,self.ENV_X).repeat(self.ENV_Y, 1), 0.0, 1.0)
        #self.current_states[:,-3] = torch.clamp(torch.linspace(2,-1,self.ENV_X).repeat(self.ENV_Y, 1), 0.0, 1.0)
        #phase = s_index/200
        #self.current_states[:,-2] = torch.full_like(self.current_states[0][0], 0.5-0.5*math.cos(phase))
        #self.current_states[:,-3] = torch.full_like(self.current_states[0][0], 0.5*math.cos(phase)+0.5)
        wig = lambda x : 1/math.exp(min(x**4.0,16.0))
        k = 1.825
        phase = s_index / 200
        self.current_states[:,-3] = wig(phase)
        self.current_states[:,-2] = wig(phase-k)
        self.current_states[:,-1] = wig(phase-2*k)
        self.current_states[:,-5] = wig(phase-3*k)
        self.current_states[:,-7] = wig(phase-4*k)

    def set_unique_control_channel(self):
        # set image target channels
        self.current_states[:,-self.target_count:,:,:] = 0.0
        for i in range(self.target_count):
            self.current_states[i*self.sims_per_image:(i+1)*self.sims_per_image][:,-(i+1)] = 1.0

    def run_pretrained(self, steps, save_all):
        self.ca_model.eval()
        with torch.no_grad():
            dat_to_vis = random.randint(0,self.cur_batch_size-1)
            for i in range(steps):
                if i%50 == 0:
                    print(f'step: {i}')
                if (save_all):
                    show_tensor(self.current_states[dat_to_vis])
                    plt.savefig(f'pretrained_output/out{self.frames_out_count:06d}.png')
                    plt.close('all')
                    show_hidden(self.current_states[dat_to_vis], 0)
                    plt.savefig(f'pretrained_output/out_hidden_{self.frames_out_count:06d}.png')
                    plt.close('all')
                    self.frames_out_count += 1
                self.sim_step()
                #self.set_experiment_control_channel(i)

    def run_sim(self, steps, run_idx, save_all):
        self.optimizer.zero_grad()
        for i in range(steps):
            reset_to_input(self.input_matsA, self.input_matsB, self.current_states)
            if (save_all):
                show_all_layers(self.current_states-self.prev_states, self.current_states)
                plt.savefig(f'output/all_figs/out_hidden_{self.frames_out_count:06d}.png')
                plt.close('all')
                self.frames_out_count += 1
            self.prev_states = self.prev_states*0.9 + 0.1*self.current_states.clone()
            self.sim_step()
            #self.set_unique_control_channel()

        loss = state_loss(self.current_states, self.final_state)
        loss.backward()
        
        '''
        with torch.no_grad():
            self.ca_model.conv1.weight.grad = self.ca_model.conv1.weight.grad/(self.ca_model.conv1.weight.grad.norm()+1e-8)
            self.ca_model.conv1.bias.grad = self.ca_model.conv1.bias.grad/(self.ca_model.conv1.bias.grad.norm()+1e-8)
            self.ca_model.conv2.weight.grad = self.ca_model.conv2.weight.grad/(self.ca_model.conv2.weight.grad.norm()+1e-8)
            self.ca_model.conv2.bias.grad = self.ca_model.conv2.bias.grad/(self.ca_model.conv2.bias.grad.norm()+1e-8)
        '''

        self.optimizer.step()
        lsv = loss.item()
        self.losses.insert(0, lsv)
        self.losses = self.losses[:100]
        print(f'running loss: {sum(self.losses)/len(self.losses)}')
        print(f'loss run {run_idx} : {lsv}')
        print(f'lr: {self.lr_schedule(run_idx)}')


    def initialize_states(self):
        self.current_states = make_initial_state(self.cur_batch_size, self.ENV_D, self.ENV_X, self.ENV_Y).to(self.device)
        self.input_matsA = make_input_fractal(self.cur_batch_size, self.input_x, self.input_y).to(self.device)
        self.input_matsB = make_input_fractal(self.cur_batch_size, self.input_x, self.input_y).to(self.device)
        self.final_state = make_final_state(self.input_matsA, self.input_matsB, self.current_states).to(self.device)
        self.prev_states = self.current_states.clone()


    def initialize_blank(self):
        self.current_states = make_initial_state(self.cur_batch_size, self.ENV_D, self.ENV_X, self.ENV_Y).to(self.device)
        self.input_matsA = torch.zeros(self.cur_batch_size,self.input_x,self.input_y).to(self.device)
        self.input_matsB = torch.zeros(self.cur_batch_size,self.input_x,self.input_y).to(self.device)
        self.final_state = make_final_state(self.input_matsA, self.input_matsB, self.current_states).to(self.device)
        self.prev_states = self.current_states.clone()
        
    def train_ca(self):
        for idx in range(self.train_steps):
            
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr_schedule(idx)
            
            #self.current_states = self.initial_state.repeat(self.cur_batch_size,1,1,1)
            self.current_states = make_initial_state(self.cur_batch_size, self.ENV_D, self.ENV_X, self.ENV_Y).to(self.device)
            self.input_matsA = make_input_fractal(self.cur_batch_size, self.input_x, self.input_y).to(self.device)
            self.input_matsB = make_input_fractal(self.cur_batch_size, self.input_x, self.input_y).to(self.device)
            self.final_state = make_final_state(self.input_matsA, self.input_matsB, self.current_states).to(self.device)
            self.prev_states = self.current_states.clone()
            self.run_sim(random.randint(self.min_sim_steps,self.max_sim_steps), idx, (idx+1)%self.evolution_interval == 0)
            if (idx % self.final_plot_interval == 0):
                show_final_target(self.input_matsA, self.input_matsB, self.current_states)
                show_tensor_surfaces(self.current_states)
                plt.savefig(f'output/out{idx:06d}.png')
                plt.close('all')
            if (idx % self.checkpoint_interval == 0):
                torch.save(self.ca_model.state_dict(), f'checkpoints/ca_model_step_{idx:06d}.pt')
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run-pretrained', dest='run_pretrained', action='store_true')

    parser.add_argument('--pretrained-path', type=str, default='ca_model_step_063500_multi_12')

    args = parser.parse_args()

    ca_sim = CASimulator()

    if args.run_pretrained:
        print('running pretained')
        ca_sim.load_pretrained(f'checkpoints/{args.pretrained_path}.pt')
        ca_sim.run_pretrained(2000, True)
    else:
        #ca_sim.load_pretrained(f'checkpoints/ca_model_mult_constant_pad.pt')
        ca_sim.train_ca()
        