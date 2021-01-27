import torch
import os
import sys

from ca_particles import CAModel

if __name__ == '__main__':
    pretrain_path = sys.argv[1]
    export_path = 'particles-vid-12-96-b.json'
    ca_model = torch.load(pretrain_path)
    # temporary compatibility patch
    #ca_model.env_d = 16
    with open(export_path, 'w') as f:
        print(ca_model.export_pytorch_ca_to_webgl_demo(), file=f)
    print(f'Done! Created file with size {os.path.getsize(export_path)}')
