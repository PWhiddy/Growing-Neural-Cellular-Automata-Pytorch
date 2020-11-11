from abc import ABC, absractmethod

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