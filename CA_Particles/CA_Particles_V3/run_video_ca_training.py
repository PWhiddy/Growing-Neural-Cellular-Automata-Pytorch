import torch
import numpy as np

from ca_particles import VideoSimulation, CAModel, CASimulation, CATrainer

if __name__ == '__main__':
    cell_dim = 16
    hidden_dim = 160
    batch_size = 24
    train_steps = 4096*9
    learning_rate = 2e-3
    lr_decay_rate = 1024*24
    model_steps_per_video_frame = 8
    step_blocks = 16
    seed = 55
    video_data_path = './p_sim_long_3_slow.npy' # https://drive.google.com/file/d/12GVWPM8YZr5Xj9U_Zdt6oBfHO-Dtz2ow/
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
        ca_sim, video_sim, sim_step_blocks_per_run=step_blocks,
        save_evolution_interval=1024, lr=learning_rate, lr_decay=lr_decay_rate,
        save_final_state_interval=16, sim_steps_per_draw=model_steps_per_video_frame,
        seed=seed, gt_reset_interval=10000000, checkpoint_path=f'checkpoints_c{cell_dim}_h{hidden_dim}'
    )
    trainer.train_standard(train_steps)
