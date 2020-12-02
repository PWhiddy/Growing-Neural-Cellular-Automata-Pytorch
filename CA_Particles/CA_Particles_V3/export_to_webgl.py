import torch
import os

from ca_particles import CAModel

if __name__ == '__main__':
    pretrain_path = 'checkpoints_512/ca_model_step_004096.pt' #'checkpoints/ca_model_step_015360.pt'
    export_path = 'multisize-particles-512-a.json'
    ca_model = torch.load(pretrain_path)
    # temporary compatibility patch
    #ca_model.env_d = 16
    with open(export_path, 'w') as f:
        print(ca_model.export_pytorch_ca_to_webgl_demo(), file=f)
    print(f'Done! Created file with size {os.path.getsize(export_path)}')
