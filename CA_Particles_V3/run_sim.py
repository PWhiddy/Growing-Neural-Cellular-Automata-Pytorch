import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from ca_particles import ParticleSimulation

def run_sim(sim, pth, name, steps=100, sim_steps_per_draw=1):
    Path(pth).mkdir(exist_ok=True)
    for i in tqdm(range(steps)):
        rendered_state = sim.draw()
        rendered_state = rendered_state[0]
        im = Image.fromarray(
            (rendered_state.detach()*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        )
        im.save(f'{pth}/{name}_step_{i:06d}.png')
        for _ in range(sim_steps_per_draw):
            sim.sim_step(0.1)

if __name__ == '__main__':
    run_sim(ParticleSimulation(), 'sim_output', 'particles', steps=1000, sim_steps_per_draw=10)
