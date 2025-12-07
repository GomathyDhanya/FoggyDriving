def describe():
    print("""
FoggyDriving — MDP Specification

TATE SPACE  S
───────────────────────────────────────────────────────────────────────────────
Observation is a continuous vector in [0,1]^(4 + L), where L = number of lidar
beams.

State components:
  • lane_onehot (2 dims):   ego lane encoded as [1,0] or [0,1]
  • speed_norm:             (ego_speed - min_speed) / (max_speed - min_speed)
  • fog_norm:               fog_level / max_fog_level
  • lidar_norm (L dims):    lidar distances normalized by max visible range

Raw state variables maintained by the environment:
  • ego_lane ∈ {0,1}
  • ego_speed ∈ [min_speed, max_speed]
  • fog ∈ {0 … max_fog_levels}
  • cars: list of dynamic objects, each with:
        lane, dist, speed, desired_speed
  • step_count
  • distance travelled


ACTION SPACE  A
───────────────────────────────────────────────────────────────────────────────
Discrete(5):

  0 = maintain speed
  1 = accelerate (speed += 1, clipped to max_speed)
  2 = decelerate      (speed -= 1, clipped to min_speed)
  3 = lane left  (lane = max(0, lane-1))
  4 = lane right (lane = min(num_lanes-1, lane+1))


TRANSITION DYNAMICS  
───────────────────────────────────────────────────────────────────────────────
Stochastic transitions due to:
  • random fog changes: with probability 0.2 each step, fog resets randomly
  • traffic generation:
        - cars spawned probabilistically if enough free space exists
        - cars removed when outside the visible window
  • IDM acceleration model for car-following dynamics
  • MOBIL lane-change logic for non-ego cars
  • noisy lidar readings (Gaussian multiplicative noise)
  • random initial traffic and speeds

Ego dynamics:
  • lane updated based on action
  • ego_speed updated based on action
  • ego perceived motion = other cars move relative to ego speed

Lidar:
  • L beams with angles in [-π/4, π/4]
  • range limited by fog: max_range_by_fog[fog_level]
  • noisy beam intersection with car bounding boxes


INITIAL STATE DISTRIBUTION  d₀
───────────────────────────────────────────────────────────────────────────────
On reset:
  • ego_lane sampled uniformly from {0,1}
  • ego_speed initialized to midpoint of [min_speed, max_speed]
  • fog sampled uniformly from {0 … max_fog_levels}
  • cars:
        - 5 to 9 randomly positioned cars
        - speeds sampled in [min_speed, max_speed-1]
        - desired speeds sampled up to max_speed
  • 3 additional slow-moving cars spawned in ego lane
  • distance = 0, step_count = 0


REWARD FUNCTION  R
───────────────────────────────────────────────────────────────────────────────
reward = ego_speed'

Penalty for collision:
      -50 added on top of the normal step reward

Bonus for reaching max episode length without crashing:
      +100 at the end of a non-terminal truncation

Interpretation:
  • incentivizes high speed
  • heavily penalizes collisions
  • encourages surviving long episodes


EPISODE TERMINATION
───────────────────────────────────────────────────────────────────────────────
• terminated = True   if collision with another car
• truncated = True    if step_count reaches max_steps


DISCOUNT FACTOR
───────────────────────────────────────────────────────────────────────────────
γ = 0.99

""")
