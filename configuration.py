import yaml
import simulation
import numpy as np


class Configuration(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Configuration, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        with open('configuration.yaml', 'r') as config:
            self.__config_file = yaml.safe_load(config)

    def get_star(self):
        return self._build_planetesimal(self.__config_file['simulation']['star'],
                                        self.__config_file['simulation']['speed_multiplier'])

    def get_planets(self):
        return [self._build_planetesimal(planet, self.__config_file['simulation']['speed_multiplier'])
                for planet in self.__config_file['simulation']['planets']]
    
    def get_g(self):
        return self.__config_file['simulation']['gravitational_constant']

    @staticmethod
    def _build_planetesimal(planet, speed_multiplier):
        position = np.array(planet['position'])
        velocity = speed_multiplier * np.array(planet['velocity'])
        print(planet)
        return simulation.Planetesimal(name=planet['name'], mass=planet['mass'], radius=planet['radius'],
                                       color=planet['color'], position_vector=position, velocity_vector=velocity)
