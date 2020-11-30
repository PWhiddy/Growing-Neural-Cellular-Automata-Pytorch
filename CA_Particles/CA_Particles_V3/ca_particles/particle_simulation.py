import numpy as np
import numpy.random as rnd
import itertools
import math
import tensorcanvas as tc
import torch

from ca_particles import Simulation, Particle, Walls

class ParticleSimulation(Simulation):
    '''
    Batched 2D particle system
    '''
    
    def __init__(self, particle_count=9, sim_count=1, force_strength=0.3, seed=117, particle_diameter=3.0, wall_pad=2,
                 p_rand_size=2.5, p_spacing=2.0, init_row_size=3, env_size=32, draw_device=torch.device('cpu')):
        self.particle_diameter = particle_diameter
        self.p_rand_size = p_rand_size
        self.force_strength = force_strength
        self.env_size = env_size
        self.walls = Walls(0, env_size, 0, env_size)
        self.wall_pad = wall_pad
        self.init_row_size = init_row_size
        self.part_spacing = p_spacing
        self.draw_device = draw_device
        self.simulation_count = sim_count # this is equivalent to batch size, ie run this many sims in parallel 
        self.particle_count = particle_count
        rnd.seed(seed)
        self.reset()

    def reset(self):
        self.step_count = 0
        self.draw_count = 0
        self.particle_systems = [[
            Particle(
                np.array([
                    (i%self.init_row_size+1)*self.part_spacing*self.particle_diameter,  
                    (i//self.init_row_size+1)*self.part_spacing*self.particle_diameter
                    ]),
                np.array([0.16*rnd.random()-0.08, 0.16*rnd.random()-0.08]),
                self.particle_diameter+self.p_rand_size*rnd.random()
            )
            for i in range(self.particle_count)]
            for _ in range(self.simulation_count)]

    def sim_step(self, time_step):
        for particles in self.particle_systems:
            for pair in itertools.combinations(particles, 2):
                Particle.pair_force(*pair, self.force_strength, time_step)
            for particle in particles:
                particle.apply_walls(self.walls, self.force_strength, time_step)
                particle.update(time_step)
        self.step_count += 1
        
    @staticmethod
    def hsv2rgb(c):
        '''
        iq's from https://www.shadertoy.com/view/lsS3Wc
        '''
        rgb = np.clip( np.abs(np.mod(c[0]*6.0+np.array([0.0,4.0,2.0]),6.0)-3.0)-1.0, 0.0, 1.0 )
        wht = np.array([1.0,1.0,1.0])
        return c[2] * (c[1]*rgb + (1.0-c[1])*wht)

    def draw(self):
        f_size = self.env_size+self.wall_pad*2
        canvas = torch.ones(self.simulation_count, 3, f_size, f_size, device=self.draw_device)
        # fill non-border (wall) area with black
        canvas[:, :, self.wall_pad:-self.wall_pad, self.wall_pad:-self.wall_pad] = 0.0
        for i, particles in enumerate(self.particle_systems):
            for particle in particles:
                col = np.clip(
                    ParticleSimulation.hsv2rgb([
                        math.atan2(*particle.velocity)/math.tau, 
                        3.0*np.sqrt(particle.velocity.dot(particle.velocity)),
                        0.9
                    ]),
                    0.0, 1.0)
                canvas[i] = tc.draw_circle(
                    *(particle.position-0.5+self.wall_pad), 
                    particle.radius, 
                    torch.tensor(col, device=self.draw_device), 
                    canvas[i])
        self.draw_count += 1
        return canvas