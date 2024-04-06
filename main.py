# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pygame as pg

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
class Planetesimal():
    def __init__(self, mass, position_vector, velocity_vector):
        self.mass = mass
        self.position = position_vector
        self.velocity = velocity_vector

class Simulator():
    def __init__(self, planetesimals):
        self.starting_planetesimals = planetesimals

    @classmethod
    def create_simulation_matrix(self, planetesimals=self.starting_planetesimals):
