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
    
    def __str__(self):
        return f'pos: ({self.position[0]}, {self.position[1]}) vel: ({self.velocity[0]}, {self.velocity[1]}) diameter: {self.diameter}' 
        
    @staticmethod
    def pair_force(a, b, force_strength, time_step):
        delta_p = a.position - b.position
        dist = np.sqrt(delta_p.dot(delta_p))
        if (dist != 0 and a.mass != 0 and b.mass != 0):
            # spring-like linear force with k=force_strength, l=part_size, m=1
            frc = (delta_p/dist)*force_strength*max(a.radius+b.radius-dist,0.0)
            a.velocity += time_step*frc/a.mass
            b.velocity -= time_step*frc/b.mass