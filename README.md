<div>
  <h3>FoggyDriving CLI</h3>

  <pre><code>python main.py --mode (describe | train | view) [options]</code></pre>

  <h4>Examples</h4>

  <p><strong>Describe the environment</strong></p>
  <pre><code>python main.py --mode describe</code></pre>
  <p>High-level description of the FoggyDriving environment.</p>

  <p><strong>Train a model</strong></p>
  <pre><code>python main.py --mode train --model PPO --timesteps 1000000 --path FoggyDrivingModel.zip</code></pre>
  <ul>
    <li><code>--model</code> can be: <code>PPO</code>, <code>A2C</code>, <code>DQN</code></li>
    <li><code>--timesteps</code> is an integer</li>
    <li><code>--path</code> is the location to save the trained model</li>
  </ul>

  <p><strong>View a trained model</strong></p>
  <pre><code>python main.py --mode view --model PPO --path FoggyDrivingModel.zip</code></pre>
  <ul>
    <li><code>--model</code> must match the algorithm used to train the model</li>
    <li><code>--path</code> points to the trained model file</li>
  </ul>
</div>


<div align="center">
  <h1>FoggyDriving — MDP Specification</h1>
</div>

<h2>State Space (S)</h2>

<p>
The observation is a continuous vector in <code>[0,1]^(4 + L)</code>, where <strong>L</strong> is the number of lidar beams.
</p>

<h3>State Components</h3>
<ul>
  <li><code>lane_onehot</code> (2 dims): ego lane encoded as [1,0] or [0,1]</li>
  <li><code>speed_norm</code>: normalized ego speed <em>(ego_speed − min_speed) / (max_speed − min_speed)</em></li>
  <li><code>fog_norm</code>: fog_level / max_fog_level</li>
  <li><code>lidar_norm</code> (L dims): lidar distances normalized by max visible range</li>
</ul>

<h3>Raw Internal State Variables</h3>
<ul>
  <li><code>ego_lane</code> ∈ {0,1}</li>
  <li><code>ego_speed</code> ∈ [min_speed, max_speed]</li>
  <li><code>fog</code> ∈ {0 … max_fog_levels}</li>
  <li><code>cars</code>: list of objects with fields lane, dist, speed, desired_speed</li>
  <li><code>step_count</code></li>
  <li><code>distance</code> travelled</li>
</ul>

<br>

<h2>Action Space (A)</h2>

<p>Discrete(5):</p>

<ul>
  <li><strong>0</strong> = maintain speed</li>
  <li><strong>1</strong> = accelerate (speed += 1, clipped to max_speed)</li>
  <li><strong>2</strong> = decelerate (speed -= 1, clipped to min_speed)</li>
  <li><strong>3</strong> = lane left (lane = max(0, lane − 1))</li>
  <li><strong>4</strong> = lane right (lane = min(num_lanes − 1, lane + 1))</li>
</ul>

<br>

<h2>Transition Dynamics</h2>


<h3>Stochastic Sources</h3>
<ul>
  <li>Random fog changes (20% chance per step)</li>
  <li>Probabilistic traffic generation</li>
  <li>IDM longitudinal acceleration model</li>
  <li>MOBIL lane-change logic</li>
  <li>Noisy lidar readings (Gaussian multiplicative noise)</li>
  <li>Random initial traffic configuration</li>
</ul>

<h3>Ego Dynamics</h3>
<ul>
  <li>Lane updated according to action</li>
  <li>Speed updated according to action</li>
  <li>Other cars move relative to ego speed (ego is reference frame)</li>
</ul>

<h3>Lidar Model</h3>
<ul>
  <li>Beams uniformly spaced in <code>[-π/4, π/4]</code></li>
  <li>Max range depends on fog level</li>
  <li>Intersections computed with car bounding boxes</li>
  <li>Noise injected into distance readings</li>
</ul>

<br>

<h2>Initial State Distribution (d₀)</h2>


<p>On reset:</p>

<ul>
  <li><code>ego_lane</code> sampled uniformly from {0,1}</li>
  <li><code>ego_speed</code> set to midpoint of speed range</li>
  <li><code>fog</code> sampled uniformly</li>
  <li>5–9 randomly positioned cars with variable speeds</li>
  <li>3 additional slow-moving cars placed in ego lane</li>
  <li><code>distance = 0</code>, <code>step_count = 0</code></li>
</ul>

<br>

<h2>Reward Function (R)</h2>

<p><code>reward = ego_speed'</code></p>

<h3>Events</h3>
<ul>
  <li><strong>Collision:</strong> −50 penalty in addition to normal speed reward</li>
  <li><strong>Survival until max_steps:</strong> +100 bonus</li>
</ul>

<p><strong>Interpretation:</strong></p>
<ul>
  <li>Encourages fast driving</li>
  <li>Strong penalty for crashing</li>
  <li>Rewards long, safe episodes</li>
</ul>

<br>

<h2>Episode Termination</h2>


<ul>
  <li><code>terminated = True</code> on collision</li>
  <li><code>truncated = True</code> when <code>step_count ≥ max_steps</code></li>
</ul>

<br>

<h2>Discount Factor</h2>


<p><code>γ = 0.99</code></p>

<br><br>

<div align="center">
  <sub>FoggyDriving Environment — README.md</sub>
</div>
