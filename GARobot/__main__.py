import numpy as np
import matplotlib.pyplot as plt
from robot_factory import Robot

from genetic_alg import garobot, GARobotConfig
from plot_utils import plot_obstacles

show_animation = False

def main():

    # Define goal point
    gx=10.0 
    gy=5.0

    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])
    pop_size = 30

    config = GARobotConfig()

    robots, env_config = garobot(pop_size, goal, config)

    plt.cla()
    plt.plot(goal[0], goal[1], "xb")
    plot_obstacles(env_config.obs)
    
    plt.xlim(env_config.env_range)
    plt.axis("equal")
    plt.grid(True)

    for robot in robots:
        plt.plot(robot.state[0], robot.state[1], "xr")
        plt.plot(robot.trajectory[:, 0], robot.trajectory[:, 1], "-r")

    plt.show()


if __name__ == '__main__':
    main()