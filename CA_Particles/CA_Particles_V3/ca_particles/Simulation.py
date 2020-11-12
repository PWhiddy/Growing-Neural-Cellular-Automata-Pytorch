from abc import ABC, abstractmethod

class Simulation(ABC):
    
    @abstractmethod
    def sim_step():
        '''
        Run one step of the simulation
        '''
        pass
    
    @abstractmethod
    def draw():
        '''
        Render the state of the simulation
        '''
        pass