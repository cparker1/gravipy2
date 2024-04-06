# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pygame as pg
import itertools

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
class Planetesimal():
    def __init__(self, name, mass, position_vector, velocity_vector):
        self.name = name
        self.mass = mass
        self.position = position_vector
        self.velocity = velocity_vector
        self.net_force = np.array([0, 0, 0])

    def apply_net_force(self, time_tick_duration):
        acceleration = self.net_force/self.mass
        self.velocity = self.velocity + acceleration
        self.position = self.position + time_tick_duration*self.velocity

    def reset_net_force(self):
        self.net_force = np.array([0,0,0])

    @classmethod
    def appy_net_gravity(cls, p1, p2, grav_constant):
        #Two planetesimal objects
        pos_diff = p2.position - p1.position
        # print(f"Pos Diff Vect = {pos_diff}")
        dist_2_to_1_squared = np.power(np.sum(np.power(pos_diff, 2)),2.0) #2.0 ensure np.power treats the sum as a floating type
        # print(f"Dist squared = {dist_2_to_1_squared}")
        force_2_to_1 = p1.mass*p2.mass*grav_constant*pos_diff/dist_2_to_1_squared #normalize position vector and scale force by distance at same time
        # print(f"Force {force_2_to_1}")
        p1.net_force = p1.net_force + force_2_to_1
        p2.net_force = p2.net_force - force_2_to_1

    def __repr__(self):
        ret_str = f"{self.name.upper()} - {self.mass} [mass]\n"
        ret_str += f"Pos: {self.position}\n"
        ret_str += f"Vel: {self.velocity}\n"
        ret_str += f"Net: {self.net_force}"
        return ret_str

class Simulator():
    def __init__(self, planetesimals, sim_step_duration=1, sim_name="None", gravity_constant=1):
        self.planetesimals = planetesimals
        self.sim_name = sim_name
        self.sim_step_duration = sim_step_duration
        self.gravity_constant = gravity_constant

        self.frames = []

    def apply_net_forces(self):
        for p in self.planetesimals:
            p.reset_net_force()
        for p1, p2 in itertools.combinations(self.planetesimals, 2):
            # print(f"{p1.name} <-> {p2.name}")
            Planetesimal.appy_net_gravity(p1, p2, self.gravity_constant)
        [p.apply_net_force(self.sim_step_duration) for p in self.planetesimals]
        self.frames.append(self.planetesimals.copy())


    def __repr__(self):
        ret_str = f"{self.sim_name} - Tick Dur: {self.sim_step_duration}\n"
        for p in self.planetesimals:
            ret_str += f"{p}\n"
        return ret_str

    def run_sim(self, iterations):
        print(self)
        [self.apply_net_forces() for _ in range(iterations)]
        print(self)


def get_test_sim():
    planet = Planetesimal("Ee-arth", 20, position_vector=np.array([0, 0, 0]), velocity_vector=np.array([0, 0, 0]))
    planet2 = Planetesimal("Marohs", 10, position_vector=np.array([1000, 0, 0]), velocity_vector=np.array([0, 0, 0]))

    sim = Simulator([planet, planet2], gravity_constant=100)
    return sim


if __name__ == "__main__":
    sim = get_test_sim()





