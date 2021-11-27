import numpy as np
import matplotlib.pyplot as plt
from robot_factory import Robot

from genetic_alg import garobot, GARobotConfig
from plot_utils import plot_obstacles

show_animation = False

def main():

    sx = 0
    sy = 5

    # Define goal point
    gx=10.0 
    gy=5.0

    # start position [x(m), y(m)]
    start = np.array([sx, sy])

    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])
    pop_size = 30

    # Create a configuration file 
    config = GARobotConfig()

    _, anim_ax = plt.subplots(figsize=(8,8))

    robots, env_config, gen_obj_vals = garobot(pop_size, start, goal, config, anim_ax, fixed=False)

    _, final_ax = plt.subplots()

    final_ax.cla()
    final_ax.plot(goal[0], goal[1], "xb")
    plot_obstacles(env_config.obs, final_ax)
    
    final_ax.set_xlim(env_config.env_range)
    final_ax.axis("equal")
    final_ax.grid(True)

    for robot in robots:
        final_ax.plot(robot.state[0], robot.state[1], "xr")
        final_ax.plot(robot.trajectory[:, 0], robot.trajectory[:, 1], "-r")

    plt.show()


if __name__ == '__main__':
    main()