import numpy as np
import math

class Particle:
    
    def __init__(self, xy_pos, xy_vel, particle_diameter):
        self.position = xy_pos
        self.velocity = xy_vel
        self.diameter = particle_diameter
        self.radius = 0.5*particle_diameter
        self.mass = 0.5*math.pi*self.radius**2
        self.damp = 1.0
    
    def update(self, time_step):
        # update position
        self.position += time_step*self.velocity
        
    def apply_walls(self, walls, w_force, time_step):        
        # repel walls
        self.velocity[0] -= time_step*w_force*walls.lower_x_overlap(self)
        self.velocity[0] += time_step*w_force*walls.upper_x_overlap(self)
        self.velocity[1] -= time_step*w_force*walls.lower_y_overlap(self)
        self.velocity[1] += time_step*w_force*walls.upper_y_overlap(self)
        
    @staticmethod
    def pair_force(a, b, force_strength, time_step):
        delta_p = a.position - b.position
        dist = np.sqrt(dv.dot(dv))
        if (dist != 0 and self.mass != 0):
            # spring-like linear force with k=force_strength, l=part_size, m=1
            frc = (dv/dist)*force_strength*max(a.radius+b.radius-dist,0.0)
            a.velocity += time_step*frc/mass
            b.velocity -= time_step*frc/mass