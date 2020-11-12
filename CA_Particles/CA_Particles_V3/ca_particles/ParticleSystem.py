import numpy as np
import numpy.random as rnd
import itertools
import math
import tensorcanvas as tc
import torch

import Simulation
import Particle
import Walls

class ParticleSystem(Simulation):
    
    def __init__(self, particle_count=6, force_strength=0.1, 
                 particle_diameter=5.0, p_spacing=10.0, init_dim=3, env_size=64):
        self.particles = []
        self.particle_diameter = particle_diameter
        self.force_strength = force_strength
        self.env_size = env_size
        self.walls = Walls(0, 0, env_size, env_size)
        self.sim_step = 0
        self.draw_step = 0
        self.init_part_dim = init_p_dim
        self.part_spacing = p_spacing
        self.init_parts(particle_count)

    def init_parts(self, particle_count):
        for i in range(particle_count):
            self.particles.append(
                Particle(
                    np.array([
                        (i%self.init_part_dim+1)*self.part_spacing*self.particle_diameter,  
                        (i//self.init_part_dim+1)*self.part_spacing*self.particle_diameter
                    ]),
                    np.array([0.16*rnd.random()-0.08, 0.16*rnd.random()-0.08]),
                    self.particle_diameter
                )
            )
        
    def sim_step(self, time_step):
        for pair in itertools.combinations(self.particles, 2):
            self.pair_force(*pair, self.force_strength, time_step)
        for particle in self.particles:
            particle.update(time_step)
            particle.apply_walls(self.walls, self.force_strength, time_step)
        self.sim_step += 1
    
    @staticmethod
    def hsv2rgb(c):
        '''
        iq's from https://www.shadertoy.com/view/lsS3Wc
        '''
        rgb = np.clip( np.abs(np.mod(c[0]*6.0+np.array([0.0,4.0,2.0]),6.0)-3.0)-1.0, 0.0, 1.0 )
        wht = np.array([1.0,1.0,1.0])
        return c[2] * (c[1]*rgb + (1.0-c[1])*wht)
    
    def draw(self):
        canvas = torch.zeros(3, self.env_size, self.env_size)
        for particle in self.particles:
            col = np.clip(
                ParticleSystem.hsv2rgb([
                    math.atan2(part.vel[0],particle.velocity[1])/math.tau, 
                    3.0*np.sqrt(part.vel.dot(particle.velocity)),
                    0.9
                ]),
                0.0, 1.0)
            canvas = tc.draw_circle(*particle.position, particle.radius, torch.tensor(col), canvas)

        #plt.savefig(f'part_test/t{self.draw_step:06d}.png')
        self.draw_step += 1
        return canvas