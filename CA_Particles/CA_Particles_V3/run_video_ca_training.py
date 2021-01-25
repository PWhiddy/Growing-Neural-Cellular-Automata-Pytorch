import torch
import numpy as np

from ca_particles import VideoSimulation, CAModel, CASimulation, CATrainer

if __name__ == '__main__':
    cell_dim = 12
    hidden_dim = 86
    batch_size = 1 #16
    train_steps = 4096*64
    model_steps_per_video_frame = 8
    seed = 55
    video_data_path = './p_sim_long_1.npy'
    device = torch.device('cuda')
    pretrain_path = None
    if pretrain_path == None:
        ca_model = CAModel(cell_dim, hidden_dim, device)
    else:
        ca_model = torch.load(pretrain_path)
    
    video_data = torch.tensor(
        np.load(video_data_path).astype(np.float32)/255,
        device=device
    ).permute(0, 3, 1, 2)
    
    video_sim = VideoSimulation(
        video_data,
        batch_size,
        model_steps_per_video_frame
    )

    ca_sim = CASimulation(
        ca_model, device, batch_size,
        env_size=video_data.shape[2], env_depth=cell_dim
    )

    trainer = CATrainer(
        ca_sim, video_sim, max_sim_step_blocks_per_run=1, block_increase_interval=1024,
        save_evolution_interval=1024, time_step=1.0, 
        save_final_state_interval=16, sim_steps_per_draw=model_steps_per_video_frame,
        seed=seed, gt_reset_interval=10000000, checkpoint_path=f'checkpoints_c{cell_dim}_h{hidden_dim}'
    )
    trainer.train_standard(train_steps)