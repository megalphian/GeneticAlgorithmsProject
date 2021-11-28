import numpy as np
import matplotlib.pyplot as plt
from robot_factory import Robot

from genetic_alg import garobot, GARobotConfig, RobotType
from plot_utils import plot_obstacles

import seaborn as sns

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
    config = GARobotConfig(50, 1, fixed=True, robot_type_gains=RobotType.SAFE, clutter_pct=15)

    show_animation = False

    if(show_animation):
        # Create a plotting axis for animation
        _, anim_ax = plt.subplots(figsize=(8,8))
    else:
        anim_ax = None

    robots, env_config, gen_obj_vals, ga_metric = garobot(pop_size, start, goal, config, anim_ax, show_animation)

    trajectory_fig, trajectory_ax = plt.subplots()

    trajectory_ax.cla()
    trajectory_ax.plot(goal[0], goal[1], "xb")
    plot_obstacles(env_config.obs, trajectory_ax)
    
    trajectory_ax.set_xlim(env_config.env_range)
    trajectory_ax.axis("equal")
    trajectory_ax.grid(True)

    for robot in robots:
        trajectory_ax.plot(robot.state[0], robot.state[1], "xr")
        trajectory_ax.plot(robot.trajectory[:, 0], robot.trajectory[:, 1], "-r")
    
    trajectory_fig.save('trajectory_fig.png', bbox_inches='tight')

    obj_val_fig, obj_val_ax = plt.subplots()
    obj_val_ax.bar(range(config.num_gens), gen_obj_vals)
    obj_val_ax.set_ylabel('Objective Value')
    obj_val_ax.set_xlabel('Generation')
    obj_val_ax.set_title('Objective value of robot population over generation')

    obj_val_fig.save('obj_val_fig.png', bbox_inches='tight')

    collision_fig, collision_ax = plt.subplots()
    collision_ax.bar(range(config.num_gens), ga_metric.avg_collisions)
    collision_ax.set_ylabel('Average Collisions')
    collision_ax.set_xlabel('Generation')
    collision_ax.set_title('Average collisions per robot per run over generation')

    collision_fig.save('collision_fig.png', bbox_inches='tight')

    time_steps_fig, time_steps_ax = plt.subplots()
    time_steps_ax.bar(range(config.num_gens), ga_metric.avg_time_steps)
    time_steps_ax.set_ylabel('Average Number of Time Steps')
    time_steps_ax.set_xlabel('Generation')
    time_steps_ax.set_title('Average Number of Time Steps per robot per run over generation')

    time_steps_fig.save('time_steps_fig.png', bbox_inches='tight')

    distance_travelled_fig, distance_travelled_ax = plt.subplots()
    distance_travelled_ax.bar(range(config.num_gens), ga_metric.avg_distance_travelled)
    distance_travelled_ax.set_ylabel('Average Distance Travelled')
    distance_travelled_ax.set_xlabel('Generation')
    distance_travelled_ax.set_title('Average Distance Travelled per robot per run over generation')

    distance_travelled_fig.save('distance_travelled_fig.png', bbox_inches='tight')

    reached_goal_fig, reached_goal_ax = plt.subplots()
    reached_goal_ax.bar(range(config.num_gens), ga_metric.avg_robots_reaching_goal)
    reached_goal_ax.set_ylabel('Average Number of Robots Reaching Goal')
    reached_goal_ax.set_xlabel('Generation')
    reached_goal_ax.set_title('Average Number of Robots Reaching Goal over generation')

    reached_goal_fig.save('reached_goal_fig.png', bbox_inches='tight')

    genome_dist_pd = ga_metric.get_genome_df()
    pal = sns.color_palette("colorblind")

    goal_gain_fig, goal_gain_ax = plt.subplots()
    sns.boxplot(x="Generation", y="Goal Gain", data=genome_dist_pd, palette=pal, ax=goal_gain_ax)
    goal_gain_ax.set_ylabel('Gain of Attractive Force Towards Goal')
    goal_gain_ax.set_xlabel('Generation')
    goal_gain_ax.set_title('Distribution of Population Goal Gain over generation')

    goal_gain_fig.save('goal_gain_fig.png', bbox_inches='tight')

    obstacle_gain_fig, obstacle_gain_ax = plt.subplots()
    sns.boxplot(x="Generation", y="Obstacle Gain", data=genome_dist_pd, palette=pal, ax=obstacle_gain_ax)
    obstacle_gain_ax.set_ylabel('Gain of Repulsive Force From Obstacle')
    obstacle_gain_ax.set_xlabel('Generation')
    obstacle_gain_ax.set_title('Distribution of Population Obstacle Gain over generation')

    obstacle_gain_fig.save('obstacle_gain_fig.png', bbox_inches='tight')

    obstacle_influence_fig, obstacle_influence_ax = plt.subplots()
    sns.boxplot(x="Generation", y="Obstacle Influence", data=genome_dist_pd, palette=pal, ax=obstacle_influence_ax)
    obstacle_influence_ax.set_ylabel('Radius of Obstacle Influence')
    obstacle_influence_ax.set_xlabel('Generation')
    obstacle_influence_ax.set_title('Distribution of Population Obstacle Influence over generation')

    obstacle_influence_fig.save('obstacle_influence_fig.png', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()