import torch
from ca_particles import ParticleSimulation, CAModel, CASimulation, CATrainer

if __name__ == '__main__':
    env_size = 24
    cell_dim = 16
    hidden_dim = 512
    batch_size = 32#128
    wall_pad = 2
    train_steps = 4096*4
    seed = 55
    device = torch.device('cuda')
    pretrain_path = 'checkpoints_512/ca_model_step_004096.pt' #None #'checkpoints/ca_model_step_009216.pt'
    if pretrain_path == None:
        ca_model = CAModel(cell_dim, hidden_dim, device)
    else:
        ca_model = torch.load(pretrain_path)
    ca_sim = CASimulation(
        ca_model, device, batch_size,
        env_size+2*wall_pad, env_depth=cell_dim
    )
    particle_sim = ParticleSimulation(
        sim_count=batch_size, 
        wall_pad=wall_pad, 
        env_size=env_size, 
        draw_device=device,
        particle_count=6,
        seed=seed
    )
    trainer = CATrainer(
        ca_sim, particle_sim, sim_step_blocks_per_run=8, 
        save_evolution_interval=64,
        seed=seed, gt_reset_interval=1024, checkpoint_path=f'checkpoints_c{cell_dim}_h{hidden_dim}'
    )
    trainer.train_standard(train_steps)