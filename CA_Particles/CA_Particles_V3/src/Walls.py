
class Walls:

    def __init__(self, lower_x, upper_x, lower_y, upper_y):
        self.lower_x = lower_x
        self.upper_x = upper_x
        self.lower_y = lower_y
        self.upper_y = upper_y
        
    def lower_x_overlap(self, particle):
        return self.overlap(self.lower_x+particle.position[0], particle.radius)
    
    def upper_x_overlap(self, particle):
        return self.overlap(self.upper_x-particle.position[0], particle.radius)
        
    def lower_y_overlap(self, particle):
        return self.overlap(self.lower_y+particle.position[1], particle.radius)
              
    def upper_y_overlap(self, particle):
        return self.overlap(self.upper_y-particle.position[1], particle.radius)
    
    def overlap(self, val, particle_radius):
        return min(val-particle_radius, 0.0)