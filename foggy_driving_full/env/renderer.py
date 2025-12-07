
import math

import matplotlib.pyplot as plt
import numpy as np
import imageio

class FoggyDrivingRender:

    def __init__(self, env):
        self.env = env

    def _draw_figure(self):
        env = self.env
        max_r = env.max_range_by_fog[env.fog]
        H = env.grid_height
        W = env.grid_width

        fig = plt.figure(figsize=(5, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax = fig.add_subplot(gs[0, 0])
        info_ax = fig.add_subplot(gs[0, 1])

        #road
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_facecolor("#d3d3d3")

        for x in range(W + 1):
            ax.axvline(x, color="white", linewidth=0.5, alpha=0.6)
        for y in range(H + 1):
            ax.axhline(y, color="white", linewidth=0.5, alpha=0.6)

        car_w = env.car_length
        car_l = env.car_length
        ego_y0 = 1.0
        ego_lane = env.ego_lane
        ego_cx = ego_lane + 0.5
        ego_cy = ego_y0 + car_l/2.0

        #ego
        ax.add_patch(
            plt.Rectangle(
                (ego_lane + 0.05, ego_y0),
                car_w - 0.1,
                car_l,
                color="blue",
            )
        )

        #other cars
        for c in env.cars:
            x0 = c.lane
            y0 = ego_y0 + c.dist
            max_r = env.max_range_by_fog[env.fog]
            fog_level = env.fog
            max_fog = env.max_fog_levels

            alpha_dist = 1.0 - min(c.dist / max_r, 1.0)
            if max_fog > 0:
                alpha_fog = 1.0 - (fog_level / max_fog) ** 1.2
            else:
                alpha_fog = 1.0

            alpha = alpha_dist * alpha_fog
            alpha = float(np.clip(alpha, 0.1, 1.0))

            if -2 <= y0 <= H + 2:
                ax.add_patch(
                    plt.Rectangle(
                        (x0 + 0.05, y0),
                        car_w - 0.1,
                        car_l,
                        color=(1, 0, 0, alpha),
                    )
                )
        lidar = env._lidar()

        for d, ang in zip(lidar, env.beam_angles):
            dx = math.sin(ang)
            dy = math.cos(ang)

            x_end = ego_cx + dx * d
            y_end = ego_cy + dy * d

            x_end = float(np.clip(x_end, 0, W))
            y_end = float(np.clip(y_end, 0, H))

            ax.plot([ego_cx, x_end], [ego_cy, y_end], color="yellow", linewidth=1.0, alpha=0.9)


        if env.fog > 0:

            base_alpha = 0.15
            alpha_step = 0.15
            fog_alpha = min(base_alpha + env.fog * alpha_step, 0.3)
            ax.add_patch(
                plt.Rectangle(
                    (0, 0),
                    W,
                    H,
                    color="gray",
                    alpha=fog_alpha,
                )
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Fog={env.fog}, Step={env.step_count}")

        #hud
        info_ax.axis("off")
        info_ax.set_title("Ego State", loc="left")

        ego_speed = env.ego_speed
        speed_norm = (ego_speed - env.min_speed) / (env.max_speed - env.min_speed + 1e-8)

        y = 0.95
        info_ax.text(0.05, y, f"Speed: {ego_speed:.1f}", fontsize=9); y -= 0.08
        info_ax.text(0.05, y, f"Speed norm: {speed_norm:.2f}", fontsize=8); y -= 0.08
        info_ax.text(0.05, y, f"Lane: {env.ego_lane}", fontsize=9); y -= 0.08
        info_ax.text(0.05, y, f"Fog level: {env.fog}", fontsize=9); y -= 0.08
        info_ax.text(0.05, y, f"Step: {env.step_count}", fontsize=9); y -= 0.08
        info_ax.text(0.05, y, f"Distance Travelled: {env.distance}",fontsize=9); y -= 0.12

        fig.tight_layout()
        return fig

    def render(self, mode="rgb_array"):
        fig = self._draw_figure()

        if mode == "human":
            plt.show()
            return None

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        plt.close(fig)
        return rgba

    def frame(self):
        return self.render(mode="rgb_array")

    def record_gif(self, model, gif_path="FoggyDriving.gif", max_steps=400):
        frames = []
        obs, _ = self.env.reset()
        done = False
        trunc = False
        step = 0

        while not (done or trunc) and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = self.env.step(int(action))
            frames.append(self.frame())
            step += 1

        imageio.mimsave(gif_path, frames, fps=8)
        print(f"GIF saved: {gif_path}")
