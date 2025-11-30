
<body>

  <h1>FoggyDriving</h1>
  <p>
    FoggyDriving is a toy custom reinforcement learning environment designed for training driving agents to perform
    lane following, overtaking, collision avoidance, and speed control under variable fog conditions.
  </p>

  <h2>FoggyDriving MDP Definition</h2>

  <h3>S (State Space)</h3>
  <p>
    Continuous vector in <code>[0, 1]^(4 + number_of_lidar_signals)</code>:
  </p>
  <ul>
    <li><code>lane_onehot</code> (2 dims)</li>
    <li><code>normalized speed</code></li>
    <li><code>normalized fog level</code></li>
    <li><code>normalized lidar distances</code></li>
  </ul>
  <p>
    Full state: <code>s = [lane_1hot, speed_norm, fog_norm, lidar_1, ..., lidar_n]</code>
  </p>

  <h3>A (Action Space)</h3>
  <p>
    Discrete actions <code>A = {0, 1, 2, 3, 4}</code>:
  </p>
  <ul>
    <li><code>0</code> = maintain</li>
    <li><code>1</code> = accelerate</li>
    <li><code>2</code> = brake</li>
    <li><code>3</code> = lane left</li>
    <li><code>4</code> = lane right</li>
  </ul>

  <h3>p (Transition Dynamics)</h3>
  <p>Stochastic transitions due to:</p>
  <ul>
    <li>random fog changes</li>
    <li>random obstacle car speeds and spawns</li>
    <li>noisy lidar readings</li>
  </ul>
  <p>Ego dynamics:</p>
  <ul>
    <li>lane shifts by at most &plusmn;1</li>
    <li>speed increases/decreases within <code>[v_min, v_max]</code></li>
  </ul>
  <p>
    Next state <code>s'</code> sampled from <code>p(s' | s, a)</code>.
  </p>

  <h3>d₀ (Initial State Distribution)</h3>
  <p>Randomized environment reset:</p>
  <ul>
    <li>random fog level</li>
    <li>random number of cars (5–8)</li>
    <li>random car distances and speeds</li>
    <li>ego lane = 1, ego speed = mid-range</li>
  </ul>
  <p>
    Defines distribution <code>d0(s) = P(s0 = s)</code>.
  </p>

  <h3>R (Reward Function)</h3>
  <p>
    <code>R(s, a, s')</code>:
  </p>
  <ul>
    <li><code>speed'</code> (normal step)</li>
    <li><code>speed' - 50</code> (collision)</li>
    <li><code>speed' + 100</code> (surviving full episode)</li>
  </ul>
  <p>
    Encourages fast but safe driving.
  </p>

  <h3>&gamma; (Discount Factor)</h3>
  <p>
    <code>gamma = 0.99</code>
  </p>

</body>
</html>
