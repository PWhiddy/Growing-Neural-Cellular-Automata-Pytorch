import torch
from ca_particles import ParticleSimulation, CAModel, CASimulation, CATrainer

if __name__ == '__main__':
    env_size = 24
    cell_dim = 16
    hidden_dim = 96
    batch_size = 24
    wall_pad = 2
    train_steps = 4096*4
    device = torch.device('cuda')
    ca_model = CAModel(cell_dim, hidden_dim, device)
    ca_sim = CASimulation(
        ca_model, device, batch_size,
        env_size+2*wall_pad, env_depth=cell_dim
    )
    particle_sim = ParticleSimulation(sim_count=batch_size, 
        wall_pad=wall_pad, 
        env_size=env_size, 
        draw_device=device,
        particle_count=6
    )
    trainer = CATrainer(ca_sim, particle_sim, max_sim_step_blocks_per_run=8, save_evolution_interval=64, time_step=1.0)
    trainer.train_standard(train_steps)