'''
Date: 2023-01-31 22:23:17
Description: 
    OpenAI Gym interface for CARLA simulator. 
    Handles sensor data, vehicle control, and scenario management.
'''

import random
import numpy as np
import pygame
from skimage.transform import resize
import gym
from gym import spaces
import carla

from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import (
    display_to_rgb,
    rgb_to_display_surface,
    get_lane_dis,
    get_pos,
    get_preview_lane_dis
)
from safebench.scenario.scenario_definition.route_scenario import RouteScenario
from safebench.scenario.scenario_definition.perception_scenario import PerceptionScenario
from safebench.scenario.scenario_definition.scenic_scenario import ScenicScenario
from safebench.scenario.scenario_manager.scenario_manager import ScenarioManager
from safebench.scenario.tools.route_manipulation import interpolate_trajectory


class CarlaEnv(gym.Env):
    """
    An OpenAI-gym style interface for CARLA simulator.
    Coordinates sensors, ego-vehicle control, and scenario execution.
    """

    def __init__(self, env_params, birdeye_render=None, display=None, world=None, logger=None):
        assert world is not None, "The CARLA world passed into CarlaEnv is None"

        self.config = None
        self.world = world
        self.display = display
        self.logger = logger
        self.birdeye_render = birdeye_render

        # Tracking simulation steps
        self.reset_step = 0
        self.total_step = 0
        self.is_running = True
        self.env_id = None
        self.ego_vehicle = None
        self.auto_ego = env_params['auto_ego']

        # Sensor variables
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.lidar_data = None
        self.lidar_height = 2.1
        self.angle = 0

        # Scenario management
        use_scenic = True if env_params['scenario_category'] == 'scenic' else False
        self.scenario_manager = ScenarioManager(self.logger, use_scenic=use_scenic)

        # Rendering and observation parameters
        self.display_size = env_params['display_size']
        self.obs_range = env_params['obs_range']
        self.d_behind = env_params['d_behind']
        self.disable_lidar = env_params['disable_lidar']

        # Environment constraints and thresholds
        self.max_past_step = env_params['max_past_step']
        self.max_episode_step = env_params['max_episode_step']
        self.max_waypt = env_params['max_waypt']
        self.lidar_bin = env_params['lidar_bin']
        self.out_lane_thres = env_params['out_lane_thres']
        self.desired_speed = env_params['desired_speed']
        self.acc_max = env_params['continuous_accel_range'][1]
        self.steering_max = env_params['continuous_steer_range'][1]

        self.ROOT_DIR = env_params['ROOT_DIR']
        self.scenario_category = env_params['scenario_category']
        self.warm_up_steps = env_params['warm_up_steps']

        # Define Observation Space based on scenario type
        if self.scenario_category in ['planning', 'scenic']:
            self.obs_size = int(self.obs_range / self.lidar_bin)
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                # State: [lateral_distance, delta_yaw, speed, vehicle_in_front]
                'state': spaces.Box(np.array([-2, -1, -5, 0], dtype=np.float32),
                                    np.array([2, 1, 30, 1], dtype=np.float32), dtype=np.float32)
            }
        elif self.scenario_category == 'perception':
            self.obs_size = env_params['image_sz']
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            }
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        self.observation_space = spaces.Dict(observation_space_dict)

        # Define Action Space (Discrete or Continuous)
        self.discrete = env_params['discrete']
        self.discrete_act = [env_params['discrete_acc'], env_params['discrete_steer']]
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            # Box space for continuous [acceleration, steering]
            self.action_space = spaces.Box(np.array([-1, -1], dtype=np.float32), 
                                           np.array([1, 1], dtype=np.float32), dtype=np.float32)

    def _create_sensors(self):
        """Initializes blueprints for sensors (Collision, LiDAR, RGB Camera)."""
        self.collision_hist_l = 1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        
        if self.scenario_category != 'perception':
            self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
            self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            self.lidar_bp.set_attribute('channels', '16')
            self.lidar_bp.set_attribute('range', '1000')

        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        self.camera_bp.set_attribute('sensor_tick', '0.02')

    def _create_scenario(self, config, env_id):
        self.logger.log(f">> Loading scenario data id: {config.data_id}")
        if self.scenario_category == 'perception':
            scenario = PerceptionScenario(world=self.world, config=config, ROOT_DIR=self.ROOT_DIR, ego_id=env_id, logger=self.logger)
        elif self.scenario_category == 'planning':
            scenario = RouteScenario(world=self.world, config=config, ego_id=env_id, max_running_step=self.max_episode_step, logger=self.logger)
        elif self.scenario_category == 'scenic':
            scenario = ScenicScenario(world=self.world, config=config, ego_id=env_id, max_running_step=self.max_episode_step, logger=self.logger)
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_manager.load_scenario(scenario)

    def reset(self, config, env_id, scenario_init_action):
        """Resets the environment for a new episode."""
        self.config = config
        self.env_id = env_id

        self._create_sensors()
        self._create_scenario(config, env_id)
        self.scenario_manager.run_scenario(scenario_init_action)
        self._attach_sensor()

        # Path planning for the ego vehicle
        self.route_waypoints = self._parse_route(config)
        self.routeplanner = RoutePlanner(self.ego_vehicle, self.max_waypt, self.route_waypoints)
        self.waypoints, _, _, _, _, self.vehicle_front, = self.routeplanner.run_step()

        # Initialize polygon tracking for visualization
        self.vehicle_polygons = [self._get_actor_polygons('vehicle.*')]
        self.walker_polygons = [self._get_actor_polygons('walker.*')]

        # Update timing
        self.time_step = 0
        self.reset_step += 1

        # Apply settings and warm up simulation
        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        for _ in range(self.warm_up_steps):
            self.world.tick()
        return self._get_obs(), self._get_info()

    def _attach_sensor(self):
        """Spawns sensor actors and attaches callback listeners."""
        # Collision Sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: self._get_collision_hist(event))
        self.collision_hist = []

        # LiDAR Sensor
        if self.scenario_category != 'perception' and not self.disable_lidar:
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego_vehicle)
            self.lidar_sensor.listen(lambda data: self._set_lidar_data(data))

        # RGB Camera Sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)
        self.camera_sensor.listen(lambda data: self._set_camera_img(data))

    def _get_collision_hist(self, event):
        impulse = event.normal_impulse
        intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_hist.append(intensity)
        if len(self.collision_hist) > self.collision_hist_l:
            self.collision_hist.pop(0)

    def _set_lidar_data(self, data):
        self.lidar_data = data

    def _set_camera_img(self, data):
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3] # Remove alpha channel
        array = array[:, :, ::-1] # BGR to RGB
        self.camera_img = array

    def step_before_tick(self, ego_action, scenario_action):
        """Applies agent actions to the ego vehicle before the simulator ticks."""
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
                # Update scenario evaluation
                if self.scenario_category == 'perception':
                    assert isinstance(ego_action, dict), 'Action must be a dict for Perception scenarios'
                    world_2_camera = np.array(self.camera_sensor.get_transform().get_inverse_matrix())
                    fov = self.camera_bp.get_attribute('fov').as_float()
                    self.scenario_manager.background_scenario.evaluate(ego_action, world_2_camera, self.obs_size, self.obs_size, fov, self.camera_img)
                    ego_action = ego_action['ego_action']

                self.scenario_manager.get_update(timestamp, scenario_action)
                self.is_running = self.scenario_manager._running

                # Convert RL action to CARLA vehicle control
                if not self.auto_ego:
                    if self.discrete:
                        acc = self.discrete_act[0][ego_action // self.n_steer]
                        steer = self.discrete_act[1][ego_action % self.n_steer]
                    else:
                        acc, steer = ego_action[0], ego_action[1]

                    # Scale and clip actions
                    acc = np.clip(acc * self.acc_max, -self.acc_max, self.acc_max)
                    steer = np.clip(steer * self.steering_max, -self.steering_max, self.steering_max)

                    # Determine Throttle vs Brake
                    if acc > 0:
                        throttle, brake = np.clip(acc / 3, 0, 1), 0
                    else:
                        throttle, brake = 0, np.clip(-acc / 8, 0, 1)

                    act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
                    self.ego_vehicle.apply_control(act)
            else:
                self.logger.log('>> Snapshot failed!', color='red')
                raise Exception()
        else:
            self.logger.log('>> CARLA world is not defined!', color='red')
            raise Exception()

    def step_after_tick(self):
        """Processes sensor results and environment state after the simulator ticks."""
        # Update polygons for actors
        self.vehicle_polygons.append(self._get_actor_polygons('vehicle.*'))
        if len(self.vehicle_polygons) > self.max_past_step: self.vehicle_polygons.pop(0)

        self.walker_polygons.append(self._get_actor_polygons('walker.*'))
        if len(self.walker_polygons) > self.max_past_step: self.walker_polygons.pop(0)

        # Update trajectories/velocities
        vehicle_info = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories.append(vehicle_info[0])
        self.vehicle_velocities.append(vehicle_info[3])
        # ... (similar logic for popping old steps)

        # Refresh waypoints
        self.waypoints, _, _, _, _, self.vehicle_front, = self.routeplanner.run_step()

        self.time_step += 1
        self.total_step += 1

        return (self._get_obs(), self._get_reward(), self._terminal(), self._get_info())

    def _calcul_orientation(self):
        """Calculates the angle difference between the vehicle direction and the road direction."""
        x, y = get_pos(self.ego_vehicle)
        wpp = self.waypoints
        dists = [(i, (x - wp[0])**2 + (y - wp[1])**2) for i, wp in enumerate(wpp)]
        m_dis = min(dists, key=lambda t: t[1])[0]
        
        a = self.waypoints[m_dis]
        a_n = self.waypoints[min(m_dis + 1, len(self.waypoints) - 1)]
        
        road_dir = np.array([a_n[0] - a[0], a_n[1] - a[1]])
        ego_dir = np.array([x - a[0], y - a[1]])
        
        road_dir /= (np.linalg.norm(road_dir) + 1e-6)
        ego_dir /= (np.linalg.norm(ego_dir) + 1e-6)
        self.angle = np.arctan2(np.cross(road_dir, ego_dir), np.dot(road_dir, ego_dir))

    def _get_walker_front(self, detection_range=15.0):
        """Checks if there are pedestrians in front of the vehicle."""
        ego_trans = self.ego_vehicle.get_transform()
        ego_loc = ego_trans.location
        ego_fwd = ego_trans.get_forward_vector()
        walkers = self.world.get_actors().filter('walker.*')

        for walker in walkers:
            w_loc = walker.get_location()
            if w_loc.distance(ego_loc) < detection_range:
                vec_to_w = w_loc - ego_loc
                dot = ego_fwd.x * vec_to_w.x + ego_fwd.y * vec_to_w.y
                if dot > 0: # Check if walker is in front of the vehicle
                    return 1.0
        return 0.0

    def _get_obs(self):
        """Aggregates sensor data and state information into a dictionary for the agent."""
        ego_trans = self.ego_vehicle.get_transform()
        ego_x, ego_y = ego_trans.location.x, ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        yaw_vec = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
        delta_yaw = np.arcsin(np.cross(w, yaw_vec))

        v = self.ego_vehicle.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        
        if not self.vehicle_front:
            self.vehicle_front = self._get_walker_front()
            
        # State used by the FM/SAC Agent: [lat_dis, yaw_error, speed, obstacle_ahead]
        state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front])

        # Render Bird-eye view if not in perception mode
        if self.scenario_category != 'perception':
            self.birdeye_render.set_hero(self.ego_vehicle, self.ego_vehicle.id)
            birdeye_surface = self.birdeye_render.render(['roadmap', 'actors', 'waypoints'])
            birdeye_surface = pygame.surfarray.array3d(birdeye_surface)
            birdeye = display_to_rgb(birdeye_surface, self.obs_size)

            # Process LiDAR point cloud into a 2D histogram image
            lidar = None
            if not self.disable_lidar:
                # ... (Point cloud binning logic)
                pass 

            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            
            # Blit to display surface for real-time monitoring
            self.display.blit(rgb_to_display_surface(birdeye, self.display_size), (0, self.env_id * self.display_size))
            self.display.blit(rgb_to_display_surface(camera, self.display_size), (self.display_size * 2, self.env_id * self.display_size))

            obs = {
                'camera': camera.astype(np.uint8),
                'lidar': lidar.astype(np.uint8) if lidar is not None else None,
                'birdeye': birdeye.astype(np.uint8),
                'state': state.astype(np.float32),
            }
        else:
            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            obs = {'camera': camera.astype(np.uint8), 'state': state.astype(np.float32)}
            
        return obs

    def _get_reward(self):
        """Calculates the scalar reward for the current step."""
        self._calcul_orientation()
        r_collision = -1 if len(self.collision_hist) > 0 else 0
        r_steer = -self.ego_vehicle.get_control().steer ** 2
        
        ego_x, ego_y = get_pos(self.ego_vehicle)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = -1 if abs(dis) > self.out_lane_thres else 0

        v = self.ego_vehicle.get_velocity()
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w) # Longitudinal speed

        # Reward for speed tracking: encourage moving but penalize exceeding desired speed
        if lspeed_lon < self.desired_speed:
            r_fast = max(lspeed_lon, 0)
        else:
            r_fast = self.desired_speed - (lspeed_lon - self.desired_speed)
            
        # Lateral acceleration penalty (comfort/stability)
        r_lat = -abs(self.ego_vehicle.get_control().steer) * lspeed_lon ** 2

        reward = (1.0 * r_collision + 1.0 * lspeed_lon + 0.5 * r_fast + 0.5 * r_out + 1.0 * r_steer + 0.2 * r_lat)
        print(f"Speed: {lspeed_lon:.3f}, Reward: {reward:.3f}")
        return reward

    def clean_up(self):
        """Destroys actors and stops sensors to prevent memory leaks."""
        self._remove_sensor()
        if self.scenario_category != 'scenic' and self.ego_vehicle:
            self.ego_vehicle.destroy()
        self.scenario_manager.clean_up()
