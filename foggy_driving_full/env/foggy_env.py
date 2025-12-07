
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional
from .renderer import FoggyDrivingRender

class FoggyDriving(gym.Env):

    class Car:
      def __init__(self, lane, dist, speed, desired_speed= None):
          self.lane = int(lane)
          self.dist = float(dist)
          self.speed = float(speed)
          self.desired_speed = float(desired_speed if desired_speed is not None else speed)


    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self,render_mode=None,min_speed=1,max_speed=5,max_fog_levels=2,max_range_by_fog=None,lidars=9,max_steps=400,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.renderer = FoggyDrivingRender(self)

        self.grid_width = 2
        self.grid_height = 40
        self.num_lanes = 2

        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.ego_speed = float((self.min_speed + self.max_speed) / 2.0)

        self.car_length = 1.0

        #IDM parameters
        self.idm_a = 1.2      #max acceleration
        self.idm_b = 2.0      #comfortable deceleration
        self.idm_T = 1.0      #desired headway time
        self.idm_s0 = 1.0     #minimum spacing
        self.idm_delta = 4    #acceleration exponent

        #MOBIL parameters
        self.mobil_safe_brake = 3.0   #max allowed braking for follower
        self.mobil_threshold = 0.3    #incentive threshold
        self.lane_change_prob = 0.2   #probability to consider lane change per step

        #fog
        self.max_fog_levels = max_fog_levels
        self.fog_levels = [i for i in range(max_fog_levels + 1)]

        if max_range_by_fog is None:
            decay = 0.6
            self.max_range_by_fog = {
                fog: float(self.grid_height) * (decay ** fog)
                for fog in self.fog_levels
            }
        else:
            self.max_range_by_fog = max_range_by_fog

        #lidar
        self.lidars = lidars
        self.beam_angles = np.linspace(-math.pi / 4, math.pi / 4, self.lidars)

        self.max_steps = max_steps

        #action space: 0 = maintain, 1 = accelerate, 2 = decellarate, 3 = lane left, 4 = lane right
        self.action_space = spaces.Discrete(5)

        #observation: 2 lane one-hot + speed + fog + lidar readings
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4 + self.lidars,), dtype=np.float32
        )


        self.rng = np.random.RandomState()

        #spawn parameters
        self.spawn_prob_per_lane = 0.2
        self.min_spawn_gap = 5.0
        self.despawn_margin = 5.0

        self.distance = 0.0
        self.ego_lane = 1
        self.step_count = 0
        self.cars: List[FoggyDriving.Car] = []

        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)

        self.step_count = 0
        self.ego_lane = int(self.rng.randint(0, self.num_lanes))
        self.ego_speed = float((self.min_speed + self.max_speed) / 2.0)
        self.fog = int(self.rng.choice(self.fog_levels))
        self.distance = 0.0

        self.cars = []

        #initial traffic
        n_cars = int(self.rng.randint(5, 10))
        for _ in range(n_cars):
            lane = int(self.rng.randint(0, self.num_lanes))
            dist = float(self.rng.uniform(4.0, self.grid_height))
            speed = float(self.rng.uniform(self.min_speed, self.max_speed - 1))
            desired = float(self.rng.uniform(max(speed, self.min_speed + 1), self.max_speed))
            self.cars.append(self.Car(lane, dist, speed, desired))


        for _ in range(3):
            dist = float(self.rng.uniform(2.0, 4.0))
            speed = float(self.min_speed)
            desired = float(self.rng.uniform(self.min_speed + 0.5, self.max_speed - 1))
            self.cars.append(self.Car(self.ego_lane, dist, speed, desired))

        obs = self._get_obs().astype(np.float32)
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.step_count += 1

        self.distance+= self.ego_speed



        if action == 1: #accelerate
            self.ego_speed = min(self.max_speed, self.ego_speed + 1.0)
        elif action == 2: #deccelerate
            self.ego_speed = max(self.min_speed, self.ego_speed - 1.0)

        if action == 3: #lane left
            self.ego_lane = max(0, self.ego_lane - 1)
        elif action == 4: #lane right
            self.ego_lane = min(self.num_lanes - 1, self.ego_lane + 1)

        self._update_cars()

        #fog level
        if self.rng.rand() < 0.2:
            self.fog = int(self.rng.choice(self.fog_levels))

        #Collision check
        collision = False
        for c in self.cars:
            if c.lane != self.ego_lane:
                continue
            if 0.0 < c.dist < self.car_length:
                collision = True
                break

        terminated = collision
        truncated = self.step_count >= self.max_steps

        reward = float(self.ego_speed)
        if collision:
            reward -= 50.0
        if truncated and not terminated:
            reward += 100.0

        obs = self._get_obs().astype(np.float32)
        info = {"collision": collision}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.renderer.render(self.render_mode)


    def _spawn_car(self, lane, dmin, dmax):

        dist = float(self.rng.uniform(dmin, dmax))
        desired_speed = float(self.rng.uniform(self.min_speed + 1, self.max_speed))
        speed = float(np.clip(desired_speed * float(self.rng.uniform(0.6, 0.9)), self.min_speed, self.max_speed))
        return self.Car(lane, dist, speed, desired_speed)

    def _idm_accel(self, car, lead):

        v = max(self.min_speed, car.speed)
        v0 = max(self.min_speed + 1e-3, car.desired_speed)

        if lead is None:
            s = 1e6
            dv = 0.0
        else:
            #gap = distance between rear of leader and front of this car
            s = lead.dist - car.dist - self.car_length
            s = max(0.1, s)
            dv = v - lead.speed

        s_star = self.idm_s0 + v * self.idm_T + (v * dv) / (2.0 * math.sqrt(self.idm_a * self.idm_b))
        accel = self.idm_a * (1.0 - (v / v0) ** self.idm_delta - (s_star / s) ** 2)
        return float(accel)

    def _find_leader(self, car, cars_in_lane):

        leader = None
        for other in cars_in_lane:
            if other.dist > car.dist + 1e-6:
                leader = other
                break
        return leader

    def _mobil_decision(self, car, delta_lane, cars_by_lane):

        target_lane = car.lane + delta_lane
        if target_lane < 0 or target_lane >= self.num_lanes:
            return False

        #current leader in own lane
        current_leader = self._find_leader(car, cars_by_lane.get(car.lane, []))
        a_current = self._idm_accel(car, current_leader)

        #in target lane, find leader and follower
        target_cars = cars_by_lane.get(target_lane, [])

        target_leader = None
        target_follower = None
        for other in target_cars:
            if other.dist > car.dist + 1e-6:
                target_leader = other
                break
        for other in reversed(target_cars):
            if other.dist < car.dist - 1e-6:
                target_follower = other
                break


        if target_follower is not None:

            old_leader_for_follower = self._find_leader(target_follower, target_cars)

            if (car.dist > target_follower.dist and
               (old_leader_for_follower is None or car.dist < old_leader_for_follower.dist)):
                new_leader_for_follower = car
            else:
                new_leader_for_follower = old_leader_for_follower

            a_follower_new = self._idm_accel(target_follower, new_leader_for_follower)
            if a_follower_new < -self.mobil_safe_brake:
                return False


        a_target = self._idm_accel(car, target_leader)
        if (a_target - a_current) < self.mobil_threshold:
            return False

        return True

    def _update_cars(self):

        cars_by_lane = {lane: [] for lane in range(self.num_lanes)}
        for c in self.cars:
            cars_by_lane[c.lane].append(c)
        for lane in cars_by_lane:
            cars_by_lane[lane].sort(key=lambda c: c.dist)


        #lane change
        desired_lane= {c: c.lane for c in self.cars}
        for car in self.cars:
            if car.dist<3:
              continue
            if self.rng.rand() < self.lane_change_prob:
                deltas = [-1, +1]
                self.rng.shuffle(deltas)
                for d in deltas:
                    if self._mobil_decision(car, d, cars_by_lane):
                        desired_lane[car] = car.lane + d
                        break


        for car, ln in desired_lane.items():
            car.lane = ln


        cars_by_lane = {lane: [] for lane in range(self.num_lanes)}
        for c in self.cars:
            cars_by_lane[c.lane].append(c)
        for lane in cars_by_lane:
            cars_by_lane[lane].sort(key=lambda c: c.dist)

        #accelerations
        accelerations = {}
        for lane, cars in cars_by_lane.items():
            for idx, car in enumerate(cars):
                lead = cars[idx + 1] if idx + 1 < len(cars) else None
                accelerations[car] = self._idm_accel(car, lead)



        upper_limit = self.grid_height + self.despawn_margin
        lower_limit = -self.despawn_margin

        new_cars = []
        for car in self.cars:
            a = accelerations.get(car, 0.0)
            car.speed = float(np.clip(car.speed + a, self.min_speed, self.max_speed))
            rel_speed = self.ego_speed - car.speed
            car.dist -= rel_speed

            if lower_limit < car.dist < upper_limit:
                new_cars.append(car)
        self.cars = new_cars

        #new cars
        visible_top = min(self.grid_height, self.max_range_by_fog[self.fog])
        spawn_base = visible_top

        for lane in range(self.num_lanes):
            lane_cars = [c for c in self.cars if c.lane == lane and c.dist >= 0.0]
            furthest = max((c.dist for c in lane_cars), default=0.0)
            free_gap = spawn_base - furthest

            if free_gap >= self.min_spawn_gap and self.rng.rand() < self.spawn_prob_per_lane:
                dmin = spawn_base
                dmax = spawn_base + self.min_spawn_gap
                new_car = self._spawn_car(lane=lane, dmin=dmin, dmax=dmax)
                self.cars.append(new_car)


    def _lidar(self):
        max_r = self.max_range_by_fog[self.fog]
        dists = np.ones(self.lidars, dtype=np.float32) * max_r

        ego_x = self.ego_lane + 0.5
        ego_y = 0.0

        car_boxes = []
        for c in self.cars:
            x0 = c.lane
            car_boxes.append((x0, x0 + 1, c.dist, c.dist + self.car_length))

        for i, angle in enumerate(self.beam_angles):
            dx = math.sin(angle)
            dy = math.cos(angle)
            t = 0.0
            step = 0.5
            hit = False
            while t < max_r and not hit:
                t += step
                x = ego_x + dx * t
                y = ego_y + dy * t
                if x < 0 or x >= self.grid_width:
                    break
                if y < 0:
                    continue
                for (xmin, xmax, ymin, ymax) in car_boxes:
                    if xmin <= x < xmax and ymin <= y < ymax:
                        dists[i] = t
                        hit = True
                        break

        noise_scale = 0.02 * (1 + 0.03 * self.fog)
        noisy = dists * (1 + self.rng.normal(0, noise_scale, size=dists.shape))
        return np.clip(noisy, 0, max_r).astype(np.float32)

    def _get_obs(self):
        lane_onehot = np.zeros(2, dtype=np.float32)
        lane_onehot[self.ego_lane] = 1.0

        speed_norm = np.float32(
            (self.ego_speed - self.min_speed)
            / (self.max_speed - self.min_speed + 1e-8)
        )
        fog_norm = np.float32(self.fog / max(self.fog_levels))

        lidar = self._lidar()
        lidar_norm = lidar / self.max_range_by_fog[self.fog]

        obs = np.concatenate(
            [
                lane_onehot,
                np.array([speed_norm], dtype=np.float32),
                np.array([fog_norm], dtype=np.float32),
                lidar_norm,
            ]
        ).astype(np.float32)
        return obs
