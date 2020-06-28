import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
import random
import json
import base64
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
        fig, axs = plt.subplots(4,t.shape[0]//4, figsize=(8, 16))
        plt.subplots_adjust(hspace =0.02, wspace=0.02)
        for axe,batch_item in zip(axs.flatten(),t):
            axe.axis('off')
            axe.imshow(to_rgb(batch_item).cpu().detach().permute(1,2,0), interpolation='nearest')

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

def state_loss(running_state, final_state):
    return F.mse_loss(running_state[:,0:3,:,:], final_state[:,0:3,:,:])
    
class CAModel(nn.Module):
    
    def __init__(self, env_d):
        super(CAModel, self).__init__()
        self.conv1 = nn.Conv2d(env_d*3,192,1)
        self.conv2 = nn.Conv2d(192,env_d,1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)
    
class CASimulator():
    
    def __init__(self):
        self.ENV_X = 128 #32
        self.ENV_Y = 128 #32
        self.ENV_D = 16
        self.particle_count = 36 # 6
        self.init_particle_dim = 6 # 3
        self.particle_size = 5.0
        self.part_spacing = 1.4
        self.repel_strength = 0.015
        self.step_size = 1.0
        self.update_probability = 0.5
        self.cur_batch_size = 1
        self.train_steps = 64000
        self.min_sim_steps = 1
        self.max_sim_steps = 8
        self.step_increase_interval = 128
        self.updates_per_step = 8
        self.device = torch.device('cuda')
        self.ca_model = CAModel(self.ENV_D)
        self.ca_model = self.ca_model.to(self.device)

        self.optimizer = optim.Adam(self.ca_model.parameters(), lr=2e-3)
        self.frames_out_count = 0
        self.losses = []
        self.checkpoint_interval = 500
        self.final_plot_interval = 1
        self.evolution_interval = 256
        self.lr_schedule = lambda x: 3e-3*2.0**(-0.0002*x) #lambda x: 2e-3 if x<4000 else 3e-4

    def initialize_particle_sims(self):
        self.p_sims = [
            ParticleSystem(
                self.particle_count, 
                self.ENV_X, 
                self.particle_size, 
                self.repel_strength,
                self.init_particle_dim,
                self.part_spacing
            ) 
            for _ in range(self.cur_batch_size)
        ]

    def draw_states(self):
        blank = torch.zeros(self.cur_batch_size, self.ENV_D, self.ENV_X, self.ENV_Y)
        blank[:,0:3,:,:] = torch.tensor([ps.draw() for ps in self.p_sims]).permute(0,3,1,2)
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

    def run_sim(self, steps, run_idx, save_all):
        self.optimizer.zero_grad()
        for i in range(steps):
            if (save_all):
                #show_all_layers(self.current_states-self.prev_states, self.current_states)
                show_tensor_surfaces(self.current_states)
                plt.savefig(f'output/all_figs/out_hidden_{self.frames_out_count:06d}.png')
                plt.close('all')
                self.frames_out_count += 1
            self.prev_states = self.prev_states*0.9 + 0.1*self.current_states.clone()
            self.sim_step()
            #self.set_unique_control_channel()

        loss = state_loss(self.current_states, self.final_states)
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

    def train_ca(self):
        self.initialize_particle_sims()
        for idx in range(self.train_steps):
            
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr_schedule(idx)
            
            #self.current_states = self.initial_state.repeat(self.cur_batch_size,1,1,1)
            #self.initialize_particle_sims()
            self.current_states = self.draw_states().to(self.device)
            num_steps = random.randint(self.min_sim_steps,min(idx//self.step_increase_interval+1,self.max_sim_steps))*self.updates_per_step
            self.run_particles(num_steps)
            self.final_states = self.draw_states().to(self.device)
            self.prev_states = self.current_states.clone()
            self.run_sim(num_steps*2, idx, (idx+1)%self.evolution_interval == 0)
            print(f'{num_steps//self.updates_per_step} blocks, {num_steps*2} total steps\n')
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
            self.current_states = self.draw_states().to(self.device)
            self.prev_states = self.current_states.clone()
            for idx in range(steps):
                print(f'step: {idx}')
                if (idx % 8 == 0):
                    #show_all_layers(self.current_states-self.prev_states, self.current_states)
                    show_tensor_surfaces(self.current_states[0])
                    plt.savefig(f'pretrained/out_{self.frames_out_count:06d}.png')
                    plt.close('all')
                    self.frames_out_count += 1
                self.prev_states = self.prev_states*0.9 + 0.1*self.current_states.clone()
                self.sim_step()

def pack_layer(weight, bias, outputType=np.uint8):
  in_ch, out_ch = weight.shape
  assert (in_ch%4==0) 
  assert (out_ch%4==0) 
  print(f'in_ch: {in_ch} out_ch: {out_ch}')
  print(f'bias.shape {bias.shape}')
  assert (bias.shape==(out_ch,))
  weight_scale, bias_scale = 1.0, 1.0
  if outputType == np.uint8:
    weight_scale = 2.0*np.abs(weight).max()
    bias_scale = 2.0*np.abs(bias).max()
    weight = np.round((weight/weight_scale+0.5)*255)
    bias = np.round((bias/bias_scale+0.5)*255)
  packed = np.vstack([weight, bias[None,...]])
  packed = packed.reshape(in_ch+1, out_ch//4, 4)
  packed = outputType(packed)
  packed_b64 = base64.b64encode(packed.tobytes()).decode('ascii')
  return {'data_b64': packed_b64, 'in_ch': in_ch, 'out_ch': out_ch,
          'weight_scale': weight_scale, 'bias_scale': bias_scale,
          'type': outputType.__name__}

  # original version
def export_pytorch_ca_to_webgl_demo(ca, outputType=np.uint8):
  # reorder the first layer inputs to meet webgl demo perception layout
  chn = ca.ENV_D
  ca = ca.ca_model
  w1 = ca.conv1.weight.squeeze().cpu().detach().permute(1,0).numpy() #ca.weights[0][0, 0].numpy()
  w1 = w1.reshape(chn, 3, -1).transpose(1, 0, 2).reshape(3*chn, -1)
  print(f'w1 shape: {w1.shape}')

  layers = [
      pack_layer(w1, ca.conv1.bias.cpu().detach().numpy(), outputType),
      pack_layer(ca.conv2.weight.squeeze().cpu().detach().permute(1,0).numpy(), ca.conv2.bias.cpu().detach().numpy(), outputType)
  ]
  return json.dumps(layers)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run-pretrained', dest='run_pretrained', action='store_true')

    parser.add_argument('--export', dest='export', action='store_true')

    parser.add_argument('--pretrained-path', type=str, default='ca_model_pretty_g_short')

    args = parser.parse_args()

    ca_sim = CASimulator()

    if args.run_pretrained:
        print('running pretained')
        ca_sim.load_pretrained(f'checkpoints/{args.pretrained_path}.pt')
        if args.export:
            with open('../../post--growing-ca/public/webgl_models8/part_weights.json', 'w') as f:
                print(export_pytorch_ca_to_webgl_demo(ca_sim), file=f)
        else:
            ca_sim.run_pretrained(50000)
    else:
        ca_sim.train_ca()
        