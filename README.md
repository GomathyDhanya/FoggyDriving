
<body>
  <div class="container">
    <h1>FoggyDriving — MDP Specification</h1>

    <h2>STATE SPACE S</h2>
    <hr />
    <p class="section-body">
      Observation is a continuous vector in <span class="inline-math">[0, 1]^(4 + L)</span>, where
      <span class="inline-math">L</span> is the number of lidar beams.
    </p>

    <p><strong>State components:</strong></p>
    <ul>
      <li><code>lane_onehot</code> (2 dims): ego lane encoded as [1, 0] or [0, 1]</li>
      <li><code>speed_norm</code>: (ego_speed − min_speed) / (max_speed − min_speed)</li>
      <li><code>fog_norm</code>: fog_level / max_fog_level</li>
      <li><code>lidar_norm</code> (L dims): lidar distances normalized by max visible range</li>
    </ul>

    <p><strong>Raw state variables maintained by the environment:</strong></p>
    <ul>
      <li><code>ego_lane</code> ∈ {0, 1}</li>
      <li><code>ego_speed</code> ∈ [min_speed, max_speed]</li>
      <li><code>fog</code> ∈ {0 … max_fog_levels}</li>
      <li><code>cars</code>: list of dynamic objects, each with lane, dist, speed, desired_speed</li>
      <li><code>step_count</code></li>
      <li><code>distance</code> travelled</li>
    </ul>

    <h2>ACTION SPACE A</h2>
    <hr />
    <p class="section-body">
      Discrete(5):
    </p>
    <ul>
      <li>0 = maintain speed</li>
      <li>1 = accelerate (speed += 1, clipped to max_speed)</li>
      <li>2 = decelerate (speed −= 1, clipped to min_speed)</li>
      <li>3 = lane left (lane = max(0, lane − 1))</li>
      <li>4 = lane right (lane = min(num_lanes − 1, lane + 1))</li>
    </ul>

    <h2>TRANSITION DYNAMICS</h2>
    <hr />
    <p><strong>Stochastic transitions due to:</strong></p>
    <ul>
      <li>Random fog changes: with probability 0.2 each step, fog resets randomly</li>
      <li>Traffic generation:
        <ul>
          <li>Cars spawned probabilistically if enough free space exists</li>
          <li>Cars removed when outside the visible window</li>
        </ul>
      </li>
      <li>IDM acceleration model for car-following dynamics</li>
      <li>MOBIL lane-change logic for non-ego cars</li>
      <li>Noisy lidar readings (Gaussian multiplicative noise)</li>
      <li>Random initial traffic and speeds</li>
    </ul>

    <p><strong>Ego dynamics:</strong></p>
    <ul>
      <li>Lane updated based on action</li>
      <li>ego_speed updated based on action</li>
      <li>Ego perceived motion = other cars move relative to ego speed</li>
    </ul>

    <p><strong>Lidar:</strong></p>
    <ul>
      <li>L beams with angles in [−π/4, π/4]</li>
      <li>Range limited by fog: <code>max_range_by_fog[fog_level]</code></li>
      <li>Noisy beam intersection with car bounding boxes</li>
    </ul>

    <h2>INITIAL STATE DISTRIBUTION d₀</h2>
    <hr />
    <p class="section-body">
      On reset:
    </p>
    <ul>
      <li><code>ego_lane</code> sampled uniformly from {0, 1}</li>
      <li><code>ego_speed</code> initialized to midpoint of [min_speed, max_speed]</li>
      <li><code>fog</code> sampled uniformly from {0 … max_fog_levels}</li>
      <li><code>cars</code>:
        <ul>
          <li>5 to 9 randomly positioned cars</li>
          <li>Speeds sampled in [min_speed, max_speed − 1]</li>
          <li>Desired speeds sampled up to max_speed</li>
        </ul>
      </li>
      <li>3 additional slow-moving cars spawned in ego lane</li>
      <li><code>distance = 0</code>, <code>step_count = 0</code></li>
    </ul>

    <h2>REWARD FUNCTION R</h2>
    <hr />
    <p class="section-body">
      <span class="inline-math">reward = ego_speed'</span>
    </p>

    <p><strong>Penalty for collision:</strong></p>
    <ul>
      <li>−50 added on top of the normal step reward</li>
    </ul>

    <p><strong>Bonus for reaching max episode length without crashing:</strong></p>
    <ul>
      <li>+100 at the end of a non-terminal truncation</li>
    </ul>

    <p><strong>Interpretation:</strong></p>
    <ul>
      <li>Incentivizes high speed</li>
      <li>Heavily penalizes collisions</li>
      <li>Encourages surviving long episodes</li>
    </ul>

    <h2>EPISODE TERMINATION</h2>
    <hr />
    <ul>
      <li><code>terminated = True</code> if collision with another car</li>
      <li><code>truncated = True</code> if <code>step_count</code> reaches <code>max_steps</code></li>
    </ul>

    <h2>DISCOUNT FACTOR</h2>
    <hr />
    <p class="section-body">
      <span class="inline-math">&gamma; = 0.99</span>
    </p>
  </div>
</body>
<