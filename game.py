# Example file showing a basic pygame "game loop"
import numpy as np
import pygame
import main
import itertools

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 1280))
clock = pygame.time.Clock()
running = True

sim = main.get_test_sim()

class Camera():
    def __init__(self, position_vector, focus_vector):
        self.position = position_vector
        self.relative_position = position_vector
        self.focus = focus_vector
        self.fov = 90
        self.max_render_distance = 1000000000
        self.initial_position = self.position.copy()

    def move_camera(self, delta_vector):
        self.relative_position = self.relative_position + delta_vector

    def move_focus(self, delta_vector):
        self.focus = self.focus + delta_vector

    def get_camera_focus_vector(self):
        return self.focus - self.position

    def reset_position(self):
        self.position = self.initial_position.copy()

    def move_camera_to_focus(self, relative_vector):
        new_position = self.relative_position + self.focus
        self.position = new_position


class SimulationRenderer():
    def __init__(self, screen, sim_states, camera):
        self.star_systems = sim_states
        self.camera = camera
        self.screen = screen

    def render_state(cls):
        pass

    def get_size_scaling(self, distance):
        return 1

    def render_object(self, object_radius, object_position, **circle_kwargs):
        focus_vector = self.camera.get_camera_focus_vector()
        focus_unit_vector = focus_vector/np.linalg.norm(focus_vector)
        focus_distance = np.linalg.norm(focus_vector)

        object_vector = object_position - self.camera.position
        object_distance = np.linalg.norm(object_vector)

        if object_vector.dot(focus_unit_vector) < 0:
            return

        perspective_vertical_vector = np.array([0,0,1])
        screen_orientation_left = np.cross(perspective_vertical_vector, focus_unit_vector)
        screen_orientation_left = screen_orientation_left/np.linalg.norm(screen_orientation_left)
        screen_orientation_up = np.cross(screen_orientation_left, focus_unit_vector)
        screen_orientation_up = screen_orientation_up/np.linalg.norm(screen_orientation_up)
        # vertical_distance = 2*focus_distance*focus_distance*(1-np.cos(self.camera.fov*np.pi/180))
        # screen_orientation_up = self.camera.focus + np.array([0,0,vertical_distance])
        # up_focus_vector = screen_orientation_up - self.camera.position
        # screen_orientation_left = screen_orientation_left/np.linalg.norm(screen_orientation_left)
        # screen_orientation_up = screen_orientation_up/np.linalg.norm(screen_orientation_up)

        object_projection_onto_focus = object_vector.dot(focus_unit_vector)
        object_angle_from_focus = np.arccos(object_projection_onto_focus/(object_distance))
        if object_angle_from_focus > self.camera.fov/2:
            return

        object_plane_focus = object_projection_onto_focus*focus_unit_vector
        object_plane_focus_diff = object_vector - object_plane_focus
        object_plane_focus_diff = object_plane_focus_diff/np.linalg.norm(object_plane_focus_diff)


        object_size_angle = 2*np.arctan(object_radius/object_distance)
        apparent_size = screen.get_height()*object_size_angle/(np.pi*(self.camera.fov/180))
        apparent_size = 5*max([0.2, apparent_size])

        distance_from_screen_center = (screen.get_height()/2)*object_angle_from_focus*180/self.camera.fov
        top_screen_distance = distance_from_screen_center*object_plane_focus_diff.dot(screen_orientation_up)
        left_screen_distance = distance_from_screen_center*(-object_plane_focus_diff.dot(screen_orientation_left))

        apparent_position = pygame.Vector2(screen.get_width()/2 + left_screen_distance, screen.get_height() / 2 + top_screen_distance)

        pygame.draw.circle(self.screen, center=apparent_position, radius=apparent_size, **circle_kwargs)

    def render_focus(self):
        return
        self.render_object(100, self.camera.focus, color="red")

    def render_planets(self):
        all_objects = []
        for s in self.star_systems:
            all_objects.extend(s.planetesimals)
        planet_dot_products = [(p, p.position.dot(self.camera.get_camera_focus_vector())) for p in all_objects]
        # planet_dot_products = list(filter(lambda x: x[1] >= 0, planet_dot_products))
        # planet_dot_products = list(filter(lambda x: x[1] < self.camera.max_render_distance**2, planet_dot_products))
        planet_dot_products.sort(key=lambda x: x[1])
        for p, distance in planet_dot_products:
            self.render_object(p.radius, p.position, color=p.color)




player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
planet_system = main.get_test_sim()
max_distance = max([np.linalg.norm(p1.position-p2.position) for p1, p2 in itertools.combinations(planet_system.planetesimals, 2)])
camera = Camera(np.array([1000,1000,max_distance]), np.array([0,0,0]))
sim_speed = 1
sim_dt_power = 0
camera_speed = 500
trail_markers_enabled = True
axis_markers_enabled = True

def get_axis_marker_planets(centered_on=np.array([0,0,0])):
    marker_planets=[]
    for n in 1000*np.arange(-100,100,1):
        for color, direction in [("white", np.array([1,0,0])), ("white", np.array([0,1,0])), ("pink", np.array([0,0,1]))]:
            marker_planets.append(main.Planetesimal(name="",
                                                    mass=0,
                                                    radius=40,
                                                    color=color,
                                                    position_vector=centered_on + direction*n,
                                                    velocity_vector=np.array([0,0,0])))
    return marker_planets

marker_system = main.Simulator(get_axis_marker_planets(), 1, "Markers", 0)

extra_markers = main.Simulator([], 1, "Markers", 0)

try:
    iteration = 0
    while running:
        iteration += 1
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        planet_system.run_sim(1+2**sim_speed)

        if iteration%30 == 0:
            new_markers = []
            for p in planet_system.planetesimals:
                new_markers.append(main.Planetesimal(name="m", mass=0, radius=p.radius/2, color=p.color, position_vector=p.position, velocity_vector=np.array([0,0,0])))
            extra_markers.planetesimals.extend(new_markers)

        focus_planet = max(planet_system.planetesimals, key=lambda p: p.mass)
        marker_system = main.Simulator(get_axis_marker_planets(centered_on=focus_planet.position), 0, "", 0)
        camera.focus = focus_planet.position
        camera.move_camera_to_focus(camera.initial_position)


        render_systems = [planet_system]
        if axis_markers_enabled == True:
            render_systems.extend([marker_system])
        if trail_markers_enabled == True:
            # render_systems.extend([extra_markers])
            if len(extra_markers.planetesimals) > 20*len(planet_system.planetesimals):
                temp_system = main.Simulator(extra_markers.planetesimals[-20*len(planet_system.planetesimals):], 0, "", 0)
                render_systems.extend([temp_system])


        sim_renderer = SimulationRenderer(screen, render_systems, camera)

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        # RENDER YOUR GAME HERE


        sim_renderer.render_focus()
        sim_renderer.render_planets()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            camera.move_focus(camera_speed*np.array([1,0,0]))
        if keys[pygame.K_s]:
            camera.move_focus(camera_speed*np.array([-1,0,0]))
        if keys[pygame.K_a]:
            camera.move_focus(camera_speed*np.array([0,1,0]))
        if keys[pygame.K_d]:
            camera.move_focus(camera_speed*np.array([0,-1,0]))
        if keys[pygame.K_LSHIFT]:
            camera.move_focus(camera_speed*np.array([0,0,1]))
        if keys[pygame.K_LCTRL]:
            camera.move_focus(camera_speed * np.array([0,0,-1]))

        if keys[pygame.K_i]:
            camera.move_camera(camera_speed*np.array([1,0,0]))
        if keys[pygame.K_k]:
            camera.move_camera(camera_speed*np.array([-1,0,0]))
        if keys[pygame.K_j]:
            camera.move_camera(camera_speed*np.array([0,1,0]))
        if keys[pygame.K_l]:
            camera.move_camera(camera_speed*np.array([0,-1,0]))
        if keys[pygame.K_y]:
            camera.move_camera(camera_speed*np.array([0,0,1]))
        if keys[pygame.K_h]:
            camera.move_camera(camera_speed * np.array([0,0,-1]))

        if keys[pygame.K_0]:
            camera.focus = np.array([0,0,0])
            camera.reset_position()


        if keys[pygame.K_EQUALS]:
            if sim_speed < 10:
                sim_speed += 1
        if keys[pygame.K_MINUS]:
            if sim_speed > 0:
                sim_speed -= 1
        if keys[pygame.K_ESCAPE]:
            raise Exception("Quit")

        if keys[pygame.K_F9]:
            planet_system = main.get_test_sim()
            marker_system = main.Simulator(get_axis_marker_planets(), 1, "Markers", 0)

        if keys[pygame.K_PAGEUP]:
            if sim_renderer.camera.fov < 120:
                sim_renderer.camera.fov += 5

        if keys[pygame.K_PAGEDOWN]:
            if sim_renderer.camera.fov > 40:
                sim_renderer.camera.fov -= 5

        if keys[pygame.K_INSERT]:
            if sim_dt_power < 100:
                sim_dt_power += 1
                planet_system.sim_step_duration = 1+sim_dt_power
        if keys[pygame.K_DELETE]:
            if sim_dt_power > 0:
                sim_dt_power -= 1
                planet_system.sim_step_duration = 1+sim_dt_power

        if keys[pygame.K_HOME]:
            trail_markers_enabled = True
        if keys[pygame.K_END]:
            trail_markers_enabled = False

        if keys[pygame.K_BACKSPACE]:
            extra_markers.planetesimals = []

        if keys[pygame.K_SLASH]:
            axis_markers_enabled = not axis_markers_enabled

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60
except Exception as e:
    pygame.display.quit()
    pygame.quit()
    raise e