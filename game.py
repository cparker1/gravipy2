# Example file showing a basic pygame "game loop"
import numpy as np
import pygame
import simulation
import itertools

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 800))
clock = pygame.time.Clock()
running = True

sim = simulation.get_test_sim()

class Camera():
    def __init__(self, position_vector, focus_vector, screen_size):
        self.position = position_vector
        self.relative_position = position_vector
        self.focus = focus_vector
        self.fov = 90
        self.max_render_distance = 1000000000
        self.initial_position = self.position.copy()
        self.screen_size = screen_size
        self.screen_border = max(screen_size)/2

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

    def get_distance_to_object(self, object_position):
        return np.linalg.norm(object_position-self.position)

    def get_screen_coordinates_from_object_coordinates(self, object_position):

        focus_vector = self.get_camera_focus_vector()
        focus_unit_vector = focus_vector / np.linalg.norm(focus_vector)

        object_vector = object_position - self.position
        object_distance = np.linalg.norm(object_vector)

        #If the object is behind the camera, don't render it
        if object_vector.dot(focus_unit_vector) < 0:
            return None

        # Distance to hypothetical plane perpendicular to focus that intersects object
        object_projection_onto_focus = object_vector.dot(focus_unit_vector)

        #Check if the object is outside field of view and shouldn't be rendered
        object_angle_from_focus = np.arccos(object_projection_onto_focus / (object_distance))
        if object_angle_from_focus > self.fov / 2:
            return None

        #Focus vector point on the hypothetical perpendicular plane that intersects object
        object_plane_focus = object_projection_onto_focus * focus_unit_vector

        #Vector from focus point to object gives position on a hypothetical plane perpendicular to the cameras focus
        object_plane_focus_diff = object_vector - object_plane_focus
        object_plane_focus_diff = object_plane_focus_diff / np.linalg.norm(object_plane_focus_diff)

        #get a vector for always up, use it to get a vector that points left by camera perspective
        perspective_vertical_vector = np.array([0,0,1])
        screen_orientation_left = np.cross(perspective_vertical_vector, focus_unit_vector)
        screen_orientation_left = screen_orientation_left/np.linalg.norm(screen_orientation_left)

        #Now use the screen left vector to get a perpendicular screen up vector
        screen_orientation_up = np.cross(screen_orientation_left, focus_unit_vector)
        screen_orientation_up = screen_orientation_up/np.linalg.norm(screen_orientation_up)

        distance_from_screen_center = self.screen_border * object_angle_from_focus * 180 / self.fov
        top_screen_distance = distance_from_screen_center * object_plane_focus_diff.dot(screen_orientation_up)
        left_screen_distance = distance_from_screen_center * (-object_plane_focus_diff.dot(screen_orientation_left))

        return self.screen_size[0]/2 + left_screen_distance, self.screen_size[1]/2 + top_screen_distance



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

        coords = self.camera.get_screen_coordinates_from_object_coordinates(object_position)
        if coords is None:
            return
        else:
            screen_x, screen_y = coords

        object_distance = self.camera.get_distance_to_object(object_position)
        object_size_angle = 2*np.arctan(object_radius/object_distance)
        apparent_size = screen.get_height()*object_size_angle/(np.pi*(self.camera.fov/180))
        apparent_size = 5*max([0.2, apparent_size])

        apparent_position = pygame.Vector2(screen_x, screen_y)

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
        planet_dot_products.sort(key=lambda x: x[1], reverse=True)
        for p, distance in planet_dot_products:
            self.render_object(p.radius, p.position, color=p.color)





player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
planet_system = simulation.get_test_sim()
max_distance = planet_system.get_largest_distance_between_objects()
camera = Camera(np.array([1000,1000,max_distance]), np.array([0,0,0]), screen_size=screen.get_size())
sim_speed = 1
sim_dt_power = 0
camera_speed = 500
zoom_speed = 2
trail_markers_enabled = True
axis_markers_enabled = True

def get_axis_marker_planets(centered_on=np.array([0,0,0])):
    marker_planets=[]
    for n in 1000*np.arange(-100,100,1):
        for color, direction in [("white", np.array([1,0,0])), ("white", np.array([0,1,0])), ("pink", np.array([0,0,1]))]:
            marker_planets.append(simulation.Planetesimal(name="",
                                                    mass=0,
                                                    radius=40,
                                                    color=color,
                                                    position_vector=centered_on + direction*n,
                                                    velocity_vector=np.array([0,0,0])))
    return marker_planets

marker_system = simulation.Simulator(get_axis_marker_planets(), 1, "Markers", 0)

extra_markers = simulation.Simulator([], 1, "Markers", 0)

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
                new_markers.append(simulation.Planetesimal(name="m", mass=0, radius=p.radius/2, color=p.color, position_vector=p.position, velocity_vector=np.array([0,0,0])))
            extra_markers.planetesimals.extend(new_markers)

        focus_planet = max(planet_system.planetesimals, key=lambda p: p.mass)
        marker_system = simulation.Simulator(get_axis_marker_planets(centered_on=focus_planet.position), 0, "", 0)
        camera.focus = focus_planet.position
        camera.move_camera_to_focus(camera.initial_position)


        render_systems = [planet_system]
        if axis_markers_enabled == True:
            render_systems.extend([marker_system])
        if trail_markers_enabled == True:
            # render_systems.extend([extra_markers])
            if len(extra_markers.planetesimals) > 20*len(planet_system.planetesimals):
                temp_system = simulation.Simulator(extra_markers.planetesimals[-20*len(planet_system.planetesimals):], 0, "", 0)
                render_systems.extend([temp_system])


        sim_renderer = SimulationRenderer(screen, render_systems, camera)

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        # RENDER YOUR GAME HERE


        # sim_renderer.render_focus()
        sim_renderer.render_planets()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            #Zoom in
            focus_vector = camera.get_camera_focus_vector()
            distance = np.linalg.norm(focus_vector)
            unit = focus_vector/distance
            speed = zoom_speed*distance/100
            camera.move_camera(speed*unit)
        if keys[pygame.K_s]:
            # Zoom out
            focus_vector = camera.get_camera_focus_vector()
            distance = np.linalg.norm(focus_vector)
            unit = focus_vector / distance
            speed = zoom_speed * distance / 100
            camera.move_camera(-speed * unit)
        if keys[pygame.K_a]:
            #Rotate LEft
            focus_vector = camera.get_camera_focus_vector()
            distance = np.linalg.norm(focus_vector[:2])
            left = np.cross(focus_vector, np.array([0,0,1]))
            left = left/np.linalg.norm(left)
            speed = distance/50 # ~tan(2 deg) = 0.02
            camera.move_camera(speed*left)
        if keys[pygame.K_d]:
            #Rotate Right
            focus_vector = camera.get_camera_focus_vector()
            distance = np.linalg.norm(focus_vector[:2])
            left = np.cross(focus_vector, np.array([0,0,1]))
            right = -left/np.linalg.norm(left)
            speed = distance/50 # ~tan(2 deg) = 0.02
            camera.move_camera(speed*right)

        if keys[pygame.K_LSHIFT]:
            focus_vector = camera.get_camera_focus_vector()
            distance = np.linalg.norm(focus_vector)
            speed = zoom_speed * distance / 100
            camera.move_camera(speed * np.array([0,0,1]))
        if keys[pygame.K_LCTRL]:
            focus_vector = camera.get_camera_focus_vector()
            distance = np.linalg.norm(focus_vector)
            speed = zoom_speed * distance / 100
            camera.move_camera(speed * np.array([0, 0, -1]))

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

        if keys[pygame.K_TAB]:
            distance = planet_system.get_largest_distance_between_objects()
            position_unit_vector = camera.relative_position/np.linalg.norm(camera.relative_position)
            camera.relative_position = distance*position_unit_vector*1.414



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
            planet_system = simulation.get_test_sim()
            marker_system = simulation.Simulator(get_axis_marker_planets(), 1, "Markers", 0)

        if keys[pygame.K_PAGEUP]:
            if sim_renderer.camera.fov < 180:
                sim_renderer.camera.fov += 5

        if keys[pygame.K_PAGEDOWN]:
            if sim_renderer.camera.fov > 10:
                sim_renderer.camera.fov -= 5

        if keys[pygame.K_INSERT]:
            if sim_dt_power < 250:
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