# This is a sample Python script.
import random

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pygame
import itertools

import main


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

def get_random_name():
    roots = ["Wex", "Till", "Fires", "Waters", "Sandy", "Spice", "TRan", "Been", "Darli", "Peent", "glor", "Effee"]
    suffixes = ["A", "B", "S"]
    return f"{roots[np.random.randint(0, len(roots)-1)]}-{suffixes[random.randint(0, len(suffixes)-1)]}"

def get_random_color():
    return list(pygame.colordict.THECOLORS.keys())[np.random.randint(0, len(pygame.colordict.THECOLORS.keys())-1)]

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
class Planetesimal():
    def __init__(self, name, mass, radius, color, position_vector, velocity_vector):
        self.name = name
        self.mass = mass
        self.color = color
        self.radius = radius
        self.position = position_vector
        self.velocity = velocity_vector
        self.net_force = np.array([0, 0, 0])

    def apply_net_force(self, time_tick_duration):
        acceleration = self.net_force/self.mass
        self.velocity = self.velocity + acceleration*time_tick_duration
        self.position = self.position + time_tick_duration*self.velocity

    def reset_net_force(self):
        self.net_force = np.array([0,0,0])

    def get_momentum(self):
        return self.velocity*self.mass
    def set_momentum(self, new_momentum):
        self.velocity = new_momentum/self.mass

    @classmethod
    def get_distance_vector(cls, p1, p2):
        return p2.position - p1.position

    @classmethod
    def get_distance(cls, p1, p2):
        return np.linalg.norm(cls.get_distance_vector(p1, p2))

    @classmethod
    def appy_net_gravity(cls, p1, p2, grav_constant):
        #Two planetesimal objects
        pos_diff = cls.get_distance_vector(p1, p2)
        # print(f"Pos Diff Vect = {pos_diff}")
        dist_2_to_1 =np.linalg.norm(pos_diff) #2.0 ensure np.power treats the sum as a floating type
        # dist_2_to_1_squared = np.max([dist_2_to_1_squared])
        # print(f"Dist squared = {dist_2_to_1_squared}")
        direction = pos_diff/dist_2_to_1
        force_2_to_1 = p1.mass*p2.mass*grav_constant*direction/dist_2_to_1**2 #normalize position vector and scale force by distance at same time
        # print(f"Force {force_2_to_1}")
        p1.net_force = p1.net_force + force_2_to_1
        p2.net_force = p2.net_force - force_2_to_1

    @classmethod
    def explode_planets(cls, p1, p2, explosive_force=1):
        num_new_masses = 8
        system_momentum = p1.get_momentum() + p2.get_momentum()
        system_net_position = (p1.position+p2.position)/2
        system_net_radius = p1.radius**2+p2.radius**2
        new_masses = [(p1.mass + p2.mass)/(num_new_masses) for _ in range(num_new_masses-1)]
        new_planets = []
        for n, new_mass in enumerate(new_masses):
            applied_momentum = np.array([new_mass*explosive_force*(np.random.rand()-0.5) for _ in range(3)])
            new_momentum = system_momentum/(num_new_masses-1) + applied_momentum
            new_velocity= new_momentum/new_mass
            new_position = np.array([(np.random.rand()-0.5)*10*(p1.radius+p2.radius) for _ in range(3)])
            new_position = system_net_position + new_position
            new_radius = np.sqrt(system_net_radius/num_new_masses)
            p = main.Planetesimal(name=get_random_name(),
                                  mass=new_mass,
                                  radius=new_radius,
                                  color=get_random_color(),
                                  position_vector=new_position,
                                  velocity_vector=new_velocity)
            new_planets.append(p)

        #Equalize momeentum
        net_new_momentum = np.sum(np.array([p.get_momentum() for p in new_planets]), axis=0)
        new_momentum = system_momentum - net_new_momentum
        new_velocity = new_momentum/new_mass
        new_position = np.array([(np.random.rand()-0.5)*10*(p1.radius+p2.radius) for _ in range(3)])
        p = main.Planetesimal(name=get_random_name(),
                              mass=new_mass,
                              radius=new_radius,
                              color=get_random_color(),
                              position_vector=new_position,
                              velocity_vector=new_velocity)
        new_planets.append(p)
        return new_planets


    @classmethod
    def handle_collision(cls, p1, p2, breakup_mass):
        if cls.get_distance(p1, p2) < (p1.radius + p2.radius):
            if p1.mass+p2.mass > breakup_mass:
                return cls.explode_planets(p1, p2)
            new_mass = p1.mass + p2.mass
            new_momentum = p1.get_momentum() + p2.get_momentum()
            new_position = p1.position# + ((p2.position - p1.position)*p1.radius)
            new_radius = 1.414*(p1.radius+p2.radius)/2
            new_color = p1.color if p1.radius > p2.radius else p2.color
            if p1.mass > p2.mass:
                new_name=f"{p1.name}-{p2.name[-3:]}"
            else:
                new_name=f"{p2.name}-{p1.name[-3:]}"

            return [cls(name=new_name,
                       mass=new_mass,
                       radius=new_radius,
                       color=new_color,
                       position_vector=new_position,
                       velocity_vector=new_momentum/new_mass)]
        return None


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
        self.initial_momentum = self.get_net_momentum()
        self.total_mass = sum([p.mass for p in planetesimals])

        self.frames = []

    def get_net_momentum(self):
        return np.sum(np.array([p.velocity*p.mass for p in self.planetesimals]), axis=0)

    def calculate_net_forces(self):
        for p in self.planetesimals:
            p.reset_net_force()
        for p1, p2 in itertools.combinations(self.planetesimals, 2):
            # print(f"{p1.name} <-> {p2.name}")
            Planetesimal.appy_net_gravity(p1, p2, self.gravity_constant)

    def apply_net_forces(self):
        self.calculate_net_forces()
        [p.apply_net_force(self.sim_step_duration) for p in self.planetesimals]
        # self.frames.append(self.planetesimals.copy())
        # print(self.get_net_momentum())


    def handle_collisions(self):
        biggest_planet = sorted(self.planetesimals, key=lambda p: p.mass)[-1]
        if len(self.planetesimals) <= 1:
            return
        for p1, p2 in itertools.combinations(self.planetesimals, 2):
            if (p1.mass == 0) or (p2.mass == 0):
                return
            if biggest_planet not in [p1, p2]:
                breakup_mass = 0.20*(self.total_mass-max([p.mass for p in self.planetesimals]))
            else:
                breakup_mass = self.total_mass+1

            new_planets = Planetesimal.handle_collision(p1, p2, breakup_mass=breakup_mass)

            if new_planets is not None:
                # print(f"{new_planet.name}")
                # print(f"from {p1.name} and {p2.name}")
                delete_planets = [p1, p2]
                self.planetesimals = [p for p in self.planetesimals if p not in delete_planets]
                self.planetesimals.extend(new_planets)
                self.handle_collisions()




    def __repr__(self):
        ret_str = f"{self.sim_name} - Tick Dur: {self.sim_step_duration}\n"
        for p in self.planetesimals:
            ret_str += f"{p}\n"
        return ret_str

    def run_sim(self, iterations):
        # print(self)
        for _ in range(iterations):
            self.apply_net_forces()
            self.handle_collisions()
            if np.max(np.abs(self.get_net_momentum() - self.initial_momentum)) > 0.01:
                # print(self)
                break
        # print(self)


def get_test_sim(g=0.2):
    def estimate_orbital_speed(gravity_constant, parent, child):
        return np.sqrt(parent.mass*gravity_constant/Planetesimal.get_distance(parent, child))

    def set_velocity(planet, new_vector):
        planet.velocity = new_vector

    def get_orbital_velocity_for_planet(gravity_constant, star, planet):
        orbital_velocity = estimate_orbital_speed(gravity_constant, star, planet)
        up = np.array([0,0,1])
        direction = np.cross(Planetesimal.get_distance_vector(star, planet), up)
        direction = direction/np.linalg.norm(direction)
        return direction*orbital_velocity+star.velocity

    speed=0.2
    star = Planetesimal("Solus", 200000, radius=400, color="red", position_vector=np.array([0, 0, 0]), velocity_vector=speed*np.array([0, 0, 0]))
    planet1 = Planetesimal("Marohs", 1000, radius=200, color="yellow", position_vector=np.array([20000, 0, 0]), velocity_vector=speed*np.array([0, -1, 0]))
    planet2 = Planetesimal("Vunes", 1000, radius=200, color="green", position_vector=np.array([0, 20000, 0]), velocity_vector=speed*np.array([1, 0, 0]))
    planet3 = Planetesimal("Jorbitar", 4000, radius=200, color="orange", position_vector=np.array([15000, 15000, 0]), velocity_vector=speed*np.array([0.7, -0.5, 0]))
    planet4 = Planetesimal("Sarnat", 4000, radius=200, color="green", position_vector=np.array([20000, -20000, 0]),
                           velocity_vector=speed * np.array([0.7, 0.5, 0]))
    planet5 = Planetesimal("Urunn", 4000, radius=200, color="blue", position_vector=np.array([30000, 0, 0]),
                           velocity_vector=speed * np.array([1, .5, 0])/2)
    planet6 = Planetesimal("Nepato", 4000, radius=200, color="blue", position_vector=np.array([-30000, 0, 0]),
                           velocity_vector=speed * np.array([0.7, 0, 0]))
    planet7 = Planetesimal("Plarn", 4000, radius=200, color="blue", position_vector=np.array([45000, 5000, -5000]),
                           velocity_vector=speed * np.array([0.7, 0, 0]))
    planet8 = Planetesimal("Chareen", 4000, radius=200, color="blue", position_vector=np.array([-45000, -30000,-10000]),
                           velocity_vector=speed * np.array([0.7, 0, 0]))
    planet9 = Planetesimal("X-9-FFB", 4000, radius=200, color="blue", position_vector=np.array([-55000, 0, 10000]),
                           velocity_vector=speed * np.array([0.7, 0, 0]))

    planets = [planet1, planet2, planet3, planet4, planet5, planet6, planet7, planet8, planet9]
    for p in planets:
        velocity = get_orbital_velocity_for_planet(g, star, p)
        p.velocity = velocity
    planets.append(star)

    sim = Simulator(planets, gravity_constant=g)
    return sim


if __name__ == "__main__":
    sim = get_test_sim()





