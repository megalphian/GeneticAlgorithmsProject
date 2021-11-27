import numpy as np
import matplotlib.pyplot as plt
from robot_factory import Robot

from genetic_alg import garobot, GARobotConfig, RobotType
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
    config = GARobotConfig(50, 3, fixed=False, robot_type_gains=RobotType.SAFE, clutter_pct=10)

    show_animation = False

    if(show_animation):
        # Create a plotting axis for animation
        _, anim_ax = plt.subplots(figsize=(8,8))
    else:
        anim_ax = None

    robots, env_config, gen_obj_vals, metrics = garobot(pop_size, start, goal, config, anim_ax, show_animation)

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

    _, obj_val_ax = plt.subplots()
    obj_val_ax.bar(range(config.num_gens), gen_obj_vals)
    obj_val_ax.set_ylabel('Objective Value')
    obj_val_ax.set_xlabel('Generation')
    obj_val_ax.set_title('Objective value of robot population over generation')

    _, collision_ax = plt.subplots()
    collision_ax.bar(range(config.num_gens), metrics[0])
    collision_ax.set_ylabel('Average Collisions')
    collision_ax.set_xlabel('Generation')
    collision_ax.set_title('Average collisions per robot per run over generation')

    _, time_steps_ax = plt.subplots()
    time_steps_ax.bar(range(config.num_gens), metrics[1])
    time_steps_ax.set_ylabel('Average Number of Time Steps')
    time_steps_ax.set_xlabel('Generation')
    time_steps_ax.set_title('Average Number of Time Steps per robot per run over generation')

    _, distance_travelled_ax = plt.subplots()
    distance_travelled_ax.bar(range(config.num_gens), metrics[2])
    distance_travelled_ax.set_ylabel('Average Distance Travelled')
    distance_travelled_ax.set_xlabel('Generation')
    distance_travelled_ax.set_title('Average Distance Travelled per robot per run over generation')

    plt.show()


if __name__ == '__main__':
    main()