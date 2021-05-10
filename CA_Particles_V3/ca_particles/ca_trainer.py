import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime
import random
import math

class CATrainer:
    
    def __init__(self, learned_model, ground_truth_model, 
                 sim_step_blocks_per_run=4, seed=115, lr=2e-3, lr_decay=1024*1024,
                 checkpoint_interval=1024, checkpoint_path='checkpoints', 
                 sim_steps_per_draw=8, gt_reset_interval=512, time_step=1.0,
                 save_final_state_interval=4, save_evolution_interval=256):
        self.gt_model = ground_truth_model
        self.ml_model = learned_model
        self.gt_reset_interval = gt_reset_interval
        self.sim_step_blocks_per_run = sim_step_blocks_per_run
        self.sim_steps_per_draw = sim_steps_per_draw
        self.time_step = time_step
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path
        self.save_final_state_interval = save_final_state_interval
        self.save_evolution_interval = save_evolution_interval
        self.run_name = "output/{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        self.lr = lr
        self.lr_decay = lr_decay
        random.seed(seed)
        
    def train_standard(self, optim_steps):
        final_state_count = 0
        evolution_count = 0
        running_loss = 0
        optimizer = optim.Adam(self.ml_model.model.parameters(), lr=self.lr)
        r_losses = np.zeros(self.sim_step_blocks_per_run)
        running_loss = 0
        for o_i in tqdm(range(1, optim_steps)):
            
            if o_i%self.gt_reset_interval == 0:
                self.gt_model.reset()
            
            new_lrate = self.lr * (0.25 ** (o_i / self.lr_decay))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            optimizer.zero_grad()
            
            # set initial state from ground truth model
            self.ml_model.reset()
            self.ml_model.states[:, 0:3, :, :] = self.gt_model.draw()
            
            c_loss = 0
            c_losses = []
             
            for s_i in range(self.sim_step_blocks_per_run):
            
                for sub_i in range(self.sim_steps_per_draw):
                    self.gt_model.sim_step(self.time_step)
                    self.ml_model.sim_step(self.time_step)
                    if o_i%self.save_evolution_interval == 0:
                        self.save_img(self.ml_model.draw()[0], 
                                      'evolution_output', 'evo', evolution_count)
                        self.save_img(self.gt_model.draw()[0], 
                                      'evolution_output_gt', 'evo', evolution_count)
                        evolution_count += 1
                
                gt_state = self.gt_model.draw()
                ml_state = self.ml_model.draw()

                loss = F.mse_loss(ml_state, gt_state)
                loss.backward(retain_graph=True)
                c_loss += loss.item()
                c_losses.append(loss.item())
                
            optimizer.step()
            
            if o_i%self.save_final_state_interval == 0:
                self.save_img(ml_state[0], 'final_state_output', 'final_state', final_state_count)
                final_state_count += 1
            
            
            if o_i == 0:
                r_losses += np.array(c_losses)
                running_loss = c_loss
            else:
                running_loss = 0.99*running_loss + 0.01*c_loss
                r_losses = 0.99*r_losses + 0.01*np.array(c_losses)
            assert math.isclose(r_losses.sum(), running_loss)
            if (o_i % 50 == 0):
                tqdm.write(f'run {o_i}, recent loss: {running_loss:.7f}, lr: {new_lrate:.5f} \nblock losses: {r_losses}')
            if o_i%self.checkpoint_interval == 0:
                self.save_model(f'ca_model_step_{o_i:06d}')
                
    def save_img(self, t, pth, fname, i):
        pth = self.run_name + '/' + pth
        Path(pth).mkdir(exist_ok=True, parents=True)
        normed = (torch.clamp(t.detach(),0.0,1.0)*255)
        im = Image.fromarray(normed.permute(1,2,0).cpu().numpy().astype(np.uint8))
        im.save(f'{pth}/{fname}_step_{i:06d}.png')
    
    def save_model(self, fname):
        pth = self.run_name + '/' + self.checkpoint_path
        Path(pth).mkdir(exist_ok=True, parents=True)
        torch.save(self.ml_model.model, f'{pth}/{fname}.pt')
        
        