# Example file showing a basic pygame "game loop"
import numpy as np
import pygame
import simulation
import itertools

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1920, 1080))
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

    def get_camera_focus_unit_vector(self):
        return (self.get_camera_focus_vector()/self.get_camera_distance_to_focus())

    def get_camera_distance_to_focus(self):
        return np.linalg.norm(self.get_camera_focus_vector())

    def reset_position(self):
        self.position = self.initial_position.copy()

    def move_camera_to_focus(self):
        new_position = self.relative_position + self.focus
        self.position = new_position

    def get_distance_to_object(self, object_position):
        return np.linalg.norm(object_position-self.position)

    def align_camera_to_focus_and_third_object(self, third_position):
        #Rotates camera to be same distance and elevation from focus, but aligns position with the focus and another body, useful for locking camera to planet orbiting a more massive object
        camera_xy_distance_from_focus = np.linalg.norm(self.relative_position[:2])#x,y only
        third_object_to_focus = self.focus[:2] - third_position[:2]
        third_object_to_focus_norm = np.linalg.norm(third_object_to_focus)
        if np.isclose(third_object_to_focus_norm, 0):
            return
        third_object_to_focus_unit_vector = third_object_to_focus/third_object_to_focus_norm
        new_x, new_y = camera_xy_distance_from_focus*third_object_to_focus_unit_vector #
        self.relative_position = np.array([new_x, new_y, self.relative_position[2]])

    def get_screen_coordinates_from_object_coordinates(self, object_position):
        focus_unit_vector = self.get_camera_focus_unit_vector()

        object_vector = object_position - self.position
        object_distance = np.linalg.norm(object_vector)

        #If the object is behind the camera, don't render it
        if object_vector.dot(focus_unit_vector) < 0:
            return None

        # Distance to hypothetical plane perpendicular to focus that intersects object
        object_projection_onto_focus = object_vector.dot(focus_unit_vector)

        #Check if the object is outside field of view and shouldn't be rendered
        object_angle_from_focus = np.arccos(object_projection_onto_focus / (object_distance))
        if object_angle_from_focus == np.nan:
            object_angle_from_focus = 0
        if object_angle_from_focus > self.fov / 2:
            return None

        #Focus vector point on the hypothetical perpendicular plane that intersects object
        object_plane_focus = object_projection_onto_focus * focus_unit_vector

        #Vector from focus point to object gives position on a hypothetical plane perpendicular to the cameras focus
        object_plane_focus_diff = object_vector - object_plane_focus
        object_distance_to_focus_on_focus_plane = np.linalg.norm(object_plane_focus_diff)
        if np.isclose(object_distance_to_focus_on_focus_plane, 0):
            object_plane_focus_diff = np.array([0,0,0])
            object_angle_from_focus = 0
        else:
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
    def __init__(self, screen, camera):
        self.camera = camera
        self.screen = screen

        self.render_queue = []

    def render_state(self, clear_queue_after_render=True):
        self.render_queue.sort(key=lambda entry: entry[0], reverse=True)
        for entry in self.render_queue:
            distance, render_function, render_args = entry
            if len(render_args) > 0:
                render_function(*render_args)
            else:
                render_function()
        if clear_queue_after_render == True:
            self.render_queue = []


    def get_size_scaling(self, distance):
        return 1

    def render_object(self, object_radius, object_position, color):

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

        pygame.draw.circle(self.screen, center=apparent_position, radius=apparent_size, color=color)

    def render_focus(self):
        return
        self.render_object(100, self.camera.focus, color="red")

    def render_line(self, color, position_one, position_two, **lineargs):
        line_start = self.camera.get_screen_coordinates_from_object_coordinates(position_one)
        line_end = self.camera.get_screen_coordinates_from_object_coordinates(position_two)
        if None in [line_end, line_start]:
            return
        pygame.draw.line(self.screen, color, line_start, line_end, width=1)

    def render_as_planets(self, planetary_systems):
        all_objects = []
        for s in planetary_systems:
            all_objects.extend(s.planetesimals)
        planet_dot_products = [(p, self.camera.get_distance_to_object(p.position)) for p in all_objects]
        # planet_dot_products = list(filter(lambda x: x[1] >= 0, planet_dot_products))
        # planet_dot_products = list(filter(lambda x: x[1] < self.camera.max_render_distance**2, planet_dot_products))
        planet_dot_products.sort(key=lambda x: x[1], reverse=True)
        for p, distance in planet_dot_products:
            self.render_queue.append([distance, self.render_object, (p.radius, p.position, p.color)])
            # self.render_object(p.radius, p.position, color=p.color)

    def render_as_lines(self, planetary_systems):
        #Hack... assuming planet system is many instances of planets with teh same name
        rendered_planets_last_position = {}
        all_objects = []
        for s in planetary_systems:
            for p in s.planetesimals:
                if p.name not in rendered_planets_last_position.keys():
                    rendered_planets_last_position[p.name] = p.position
                else:
                    last_pos = rendered_planets_last_position[p.name]
                    this_pos = p.position
                    rendered_planets_last_position[p.name] = this_pos
                    distance = self.camera.get_distance_to_object((this_pos+last_pos)/2)
                    self.render_queue.append([distance, self.render_line, (p.color, last_pos, this_pos)])

    def render_markers(self):
        all_objects = []


    # Build render queue, sort by
    # (apparent_distance, render_function, *render_function_args)





player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
planet_system = simulation.get_test_sim()
max_distance = planet_system.get_largest_distance_between_objects()
focus_planet = planet_system.get_most_massive_planet()
camera = Camera(np.array([max_distance,max_distance,max_distance]), np.array([0,0,0]), screen_size=screen.get_size())
lock_planet_viewing_angle = False
sim_speed = 1
sim_dt_power = 100
planet_system.sim_step_duration = 1+sim_dt_power
camera_speed = 500
zoom_speed = 2
trail_markers_enabled = True
axis_markers_enabled = True
time_interval_for_capture = 1000
desired_trails = 100
background_color = "black"


def get_next_planet(this_planet, planet_system, previous=False):
    if this_planet in planet_system.planetesimals:
        idx = planet_system.planetesimals.index(this_planet)
        if previous is True:
            idx -= 1
        else:
            idx += 1
        if idx <= 0:
            idx = -1
        elif idx >= len(planet_system.planetesimals)-1:
            idx = 0
        return planet_system.planetesimals[idx]
    else:
        return get_closest_planet(this_planet, planet_system)

def get_closest_planet(this_planet, planet_system):
    return planet_system.get_object_closest_to_position(this_planet.position)


def get_axis_marker_planet_systems(centered_on=np.array([0,0,0]), range=10000):
    axis_planet_systems = []
    for color, direction in [("white", np.array([1,0,0])), ("white", np.array([0,1,0])), ("pink", np.array([0,0,1]))]:
        axis_planet_systems.append(simulation.Simulator([simulation.Planetesimal(name=str(direction), color=color, position_vector=centered_on-range*direction), simulation.Planetesimal(name=str(direction), color=color, position_vector=centered_on+range*direction)]))
    return axis_planet_systems

axis_marker_systems = get_axis_marker_planet_systems()

extra_markers = simulation.Simulator([], 1, "Markers", 0)

try:
    iteration = 0
    iteration_of_last_focus_change = 0
    time_since_last_capture = 0
    while running:
        iteration += 1
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        planet_system.run_sim(1+2**sim_speed)
        screen.fill(background_color)

        time_since_capture = planet_system.sim_time - time_since_last_capture
        if time_since_capture > time_interval_for_capture:
            time_since_last_capture = planet_system.sim_time
            new_markers = []
            for p in planet_system.planetesimals:
                new_markers.append(simulation.Planetesimal(name=p.name, mass=0, radius=p.radius/2, color=p.color, position_vector=p.position, velocity_vector=np.array([0,0,0])))
            extra_markers.planetesimals.extend(new_markers)

        most_massive_planet = planet_system.get_most_massive_planet()
        axis_marker_systems = get_axis_marker_planet_systems(centered_on=most_massive_planet.position)

        # move camera focus to nearest planet if planet no longer a thing
        if focus_planet not in planet_system.planetesimals:
            focus_planet = get_closest_planet(focus_planet, planet_system)

        camera.focus = focus_planet.position
        camera.move_camera_to_focus()
        if lock_planet_viewing_angle is True:
            camera.align_camera_to_focus_and_third_object(most_massive_planet.position)
            camera.move_camera_to_focus()



        planetary_systems = [planet_system]
        trail_length = min([desired_trails, len(extra_markers.planetesimals)])
        trail_system = simulation.Simulator(extra_markers.planetesimals[-trail_length:], 0, "", 0)
        # planetary_systems.extend([trail_system])


        sim_renderer = SimulationRenderer(screen, camera)
        # fill the screen with a color to wipe away anything from last frame


        # RENDER YOUR GAME HERE


        # sim_renderer.render_focus()
        sim_renderer.render_as_planets(planetary_systems)
        if axis_markers_enabled == True:
            sim_renderer.render_as_lines(axis_marker_systems)
        sim_renderer.render_as_lines([trail_system])
        sim_renderer.render_state()

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

        if keys[pygame.K_q]:
            #Lock angle of focus to coordinate system planet
            lock_planet_viewing_angle = True

        if keys[pygame.K_e]:
            lock_planet_viewing_angle = False


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
            axis_marker_systems = get_axis_marker_planet_systems(centered_on=most_massive_planet.position)

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
            if desired_trails < 1000:
                desired_trails += 10

        if keys[pygame.K_END]:
            desired_trails -= 10
            if desired_trails < 0:
                desired_trails = 1

        if keys[pygame.K_RIGHT]:
            if iteration - iteration_of_last_focus_change > 20:
                focus_planet = get_next_planet(focus_planet, planet_system, previous=False)
                iteration_of_last_focus_change = iteration

        if keys[pygame.K_LEFT]:
            if iteration - iteration_of_last_focus_change > 20:
                focus_planet = get_next_planet(focus_planet, planet_system, previous=True)
                iteration_of_last_focus_change = iteration

        if keys[pygame.K_DOWN]:
            focus_planet = planet_system.get_most_massive_planet()

        if keys[pygame.K_BACKSPACE]:
            extra_markers.planetesimals = []

        if keys[pygame.K_SLASH]:
            if iteration - iteration_of_last_focus_change > 20:
                iteration_of_last_focus_change = iteration
                axis_markers_enabled = not axis_markers_enabled

        if keys[pygame.K_b]:
            if iteration - iteration_of_last_focus_change > 20:
                iteration_of_last_focus_change = iteration
                if background_color == "black":
                    background_color = "white"
                else:
                    background_color = "black"

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60
except Exception as e:
    raise e
finally:
    pygame.display.quit()
    pygame.quit()
