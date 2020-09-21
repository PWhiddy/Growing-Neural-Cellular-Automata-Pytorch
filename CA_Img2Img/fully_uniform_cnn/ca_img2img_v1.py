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
from torchvision import utils
from ca_model import CAModel
#from wide_kernel_model import WideKModel
from cifar_dataset import CifarImg2ImgDataset
random.seed(2577)

def to_rgb(x):
    rgb = x[0:3,:,:] * 0.5 + 0.5
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
    
class CASimulator():
    
    def __init__(self):
        self.ENV_X = 32
        self.ENV_Y = 42
        self.ENV_D = 128
        self.hidden_dim = 128
        self.step_size = 1.0
        self.update_probability = 0.9
        self.cur_batch_size = 32
        self.batch_expander = 1
        self.epochs = 128
        self.sim_steps = 32

        self.device = torch.device('cuda')
        self.ca_model = CAModel(self.ENV_D, self.hidden_dim)
        self.ca_model = self.ca_model.to(self.device)

        cf_train = CifarImg2ImgDataset('./label_imgs/cifar-labels', train=True)
        cf_test = CifarImg2ImgDataset('./label_imgs/cifar-labels', train=False)
        self.train_data = torch.utils.data.DataLoader(cf_train, batch_size=self.cur_batch_size,
                                          shuffle=True, num_workers=2)
        self.test_data = torch.utils.data.DataLoader(cf_test, batch_size=self.cur_batch_size,
                                          shuffle=True, num_workers=2)
        self.optimizer = optim.Adam(self.ca_model.parameters(), lr=3e-4)
        self.opt_steps = 0
        self.train_frames_out_count = 0
        self.test_frames_out_count = 0
        self.losses = []
        self.writer = SummaryWriter()
        self.checkpoint_interval = 500
        self.final_plot_interval = self.batch_expander
        self.evolution_interval = 1024
        # lr decay + sigmoid warm-up for training restarts
        #self.lr_schedule = lambda x: min(x*0.0004,2e-3) #lambda x: 2e-3 if x<4000 else 3e-4

    def load_pretrained(self, path):
        self.ca_model.load_state_dict(torch.load(path))
 
    def wrap_edges(self, x):
        return F.pad(x, (2,2,2,2), 'circular', 0)

    def raw_senses(self):
        # state - (batch, depth, x, y)
        return self.wrap_edges(self.current_states)
        '''
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
        '''
        
    def sim_step(self):
        state_updates = self.ca_model(self.raw_senses())*self.step_size
        # randomly block updates to enforce
        # asynchronous communication between cells
        rand_mask = torch.rand_like(
            self.current_states[:, :1, :, :], device=self.device) < self.update_probability
        self.current_states += state_updates*(rand_mask.float())

    def initialize_states(self, initials, targets):
        self.current_states = torch.zeros([initials.shape[0], self.ENV_D, initials.shape[2], initials.shape[3]], device=self.device)
        self.current_states[:,0:3,:,:] = initials
        self.final_states = torch.zeros([initials.shape[0], self.ENV_D, initials.shape[2], initials.shape[3]], device=self.device)
        self.final_states[:,0:3, :, :] = targets

    def do_train_batch(self, run_idx, initials, targets, steps, save_all):
        self.ca_model.train()
        self.initialize_states(initials, targets)
        
        for i in range(steps):
            if (save_all):
                show_tensor_surfaces(self.current_states)
                plt.savefig(f'output/all_figs/train_out_hidden_{self.train_frames_out_count:06d}.png')
                plt.close('all')
                self.train_frames_out_count += 1
            self.sim_step()
        
        loss = state_loss(self.current_states, self.final_states)
        loss.backward()
        
        do_optim = (run_idx+1) % self.batch_expander == 0
        if do_optim:
            torch.nn.utils.clip_grad_norm_(self.ca_model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            lsv = loss.item()
            #self.losses.insert(0, lsv)
            #self.losses = self.losses[:100]
            self.writer.add_scalar('Loss/batch-train', lsv, self.opt_steps)
            #self.writer.add_scalar('Train/Loss/moving_average', sum(self.losses)/len(self.losses), run_idx)
            #self.writer.add_scalar('LearningRate/val', self.lr_schedule(self.opt_steps), run_idx)
            self.opt_steps += 1

    def do_test_batch(self, run_idx, initials, targets, steps, save_all):
        self.ca_model.eval()
        with torch.no_grad():
            self.initialize_states(initials, targets)
            for i in range(steps):
                if (save_all):
                    show_tensor_surfaces(self.current_states)
                    plt.savefig(f'output/all_figs/test_out_hidden_{self.test_frames_out_count:06d}.png')
                    plt.close('all')
                    self.test_frames_out_count += 1
                self.sim_step()
            loss = state_loss(self.current_states, self.final_states)
            lsv = loss.item()
            self.writer.add_scalar('Loss/batch-test', lsv, run_idx)


    def train_ca(self):
        for ep in range(self.epochs):

            #for g in self.optimizer.param_groups:
            #    g['lr'] = self.lr_schedule(self.opt_steps)

            for idx, (inits, targs) in enumerate(self.train_data):
                inits = inits.to(self.device)
                targs = targs.to(self.device)
                num_steps = self.sim_steps 
                # run_idx, initials, targets, steps, save_all
                self.do_train_batch(idx, inits, targs, num_steps, (idx+1)%self.evolution_interval == 0)
                if (idx % self.final_plot_interval == 0):
                    #self.writer.add_image('images', utils.make_grid(self.current_states[:,0:3,:,:]), idx)
                    show_tensor_surfaces(self.current_states)
                    plt.savefig(f'output/train_out{self.opt_steps:06d}.png')
                    plt.close('all')
                #if (idx % 150 == 0):
                #    self.writer.add_histogram('weights/conv1', self.ca_model.conv1.weight.clone().detach().cpu().numpy().flatten(), idx)
                #    self.writer.add_histogram('weights/conv2', self.ca_model.conv2.weight.clone().detach().cpu().numpy().flatten(), idx)

            for idx, (inits, targs) in enumerate(self.test_data):
                inits = inits.to(self.device)
                targs = targs.to(self.device)
                num_steps = self.sim_steps 
                # run_idx, initials, targets, steps, save_all
                self.do_test_batch(idx, inits, targs, num_steps, (idx+1)%self.evolution_interval == 0)
                if (idx % self.final_plot_interval == 0):
                    #self.writer.add_image('images', utils.make_grid(self.current_states[:,0:3,:,:]), idx)
                    show_tensor_surfaces(self.current_states)
                    plt.savefig(f'output/test_out{idx:06d}.png')
                    plt.close('all')
                torch.save(self.ca_model.state_dict(), f'checkpoints/ca_model_step_{ep:06d}.pt')


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
        #ca_sim.load_pretrained(f'checkpoints/ca_model_b2.pt')
        ca_sim.train_ca()
        