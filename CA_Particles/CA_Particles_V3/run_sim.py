from pathlib import Path
import matplotlib.pyplot as plt

from ca_particles import ParticleSystem

def run_sim(sim, pth, name, steps=100):
    Path(pth).mkdir(exist_ok=True)
    for i in range(steps):
        rendered_state = sim.draw()
        plt.imshow(rendered_state.transpose(1,2,0))
        plt.savefig(f'{pth}/{name}_step_{i:06d}.png')
        sim.sim_step()

if __name__ == '__main__':
    test_sim(ParticleSystem(), 'sim_output', 'particles', steps=1000)
