'''
Utility functions for plotting the results of GAROBOT
'''

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

def plot_robot(x, y, radius, ax):  # pragma: no cover
    circle = plt.Circle((x, y), radius, color="b")
    ax.add_artist(circle)

def plot_obstacles(obs, ax):  # pragma: no cover
    for ob in obs:
        circle = plt.Circle((ob[0], ob[1]), ob[2], color="k")
        ax.add_artist(circle)

def animate(anim_ax, env_config, goal, robots, obs):
    anim_ax.cla()
    # for stopping simulation with the esc key.
    anim_ax.figure.canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    anim_ax.plot(goal[0], goal[1], "xb")
    plot_obstacles(obs, anim_ax)

    for robot in robots:
        anim_ax.plot(robot.state[0], robot.state[1], "xr")
        plot_robot(robot.state[0], robot.state[1], robot.robot_radius, anim_ax)

    anim_ax.set_xlim([env_config.env_range[0] - 0.2, env_config.env_range[1] + 0.2])
    anim_ax.set_ylim([env_config.env_range[0] - 0.2, env_config.env_range[1] + 0.2])
    anim_ax.grid(True)
    plt.pause(0.0001)

def plot_final_results(ga_metric, goal, env_config, ga_config):
    for i in range(len(ga_metric.recorded_trajectories)):

        obs = ga_metric.recorded_obs[i]
        trajectories = ga_metric.recorded_trajectories[i]

        trajectory_fig, trajectory_ax = plt.subplots()

        trajectory_ax.cla()
        trajectory_ax.plot(goal[0], goal[1], "xg")
        plot_obstacles(obs, trajectory_ax)
        
        trajectory_ax.set_xlim(env_config.env_range)
        trajectory_ax.axis("equal")
        trajectory_ax.grid(True)

        trajectory_ax.set_title('Trajectories at the end of Generation ' + str(ga_metric.recorded_gens[i]))

        for trajectory in trajectories:
            trajectory_ax.plot(trajectory[-1, 0], trajectory[-1, 1], "ob")
            trajectory_ax.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        
        trajectory_fig.savefig('trajectory_fig_' + str(i) + '.png', bbox_inches='tight')


    obj_val_fig, obj_val_ax = plt.subplots()
    obj_val_ax.bar(range(ga_config.num_gens), ga_metric.gen_obj_vals)
    obj_val_ax.set_ylabel('Objective Value')
    obj_val_ax.set_xlabel('Generation')
    obj_val_ax.set_title('Objective value of robot population over generation')

    obj_val_fig.savefig('obj_val_fig.png', bbox_inches='tight')

    collision_fig, collision_ax = plt.subplots()
    collision_ax.bar(range(ga_config.num_gens), ga_metric.avg_collisions)
    collision_ax.set_ylabel('Average Collisions')
    collision_ax.set_xlabel('Generation')
    collision_ax.set_title('Average collisions per robot per run over generation')

    collision_fig.savefig('collision_fig.png', bbox_inches='tight')

    time_steps_fig, time_steps_ax = plt.subplots()
    time_steps_ax.bar(range(ga_config.num_gens), ga_metric.avg_time_steps)
    time_steps_ax.set_ylabel('Average Number of Time Steps')
    time_steps_ax.set_xlabel('Generation')
    time_steps_ax.set_title('Average Number of Time Steps per robot per run over generation')

    time_steps_fig.savefig('time_steps_fig.png', bbox_inches='tight')

    distance_travelled_fig, distance_travelled_ax = plt.subplots()
    distance_travelled_ax.bar(range(ga_config.num_gens), ga_metric.avg_distance_travelled)
    distance_travelled_ax.set_ylabel('Average Distance Travelled')
    distance_travelled_ax.set_xlabel('Generation')
    distance_travelled_ax.set_title('Average Distance Travelled per robot per run over generation')

    distance_travelled_fig.savefig('distance_travelled_fig.png', bbox_inches='tight')

    reached_goal_fig, reached_goal_ax = plt.subplots()
    reached_goal_ax.bar(range(ga_config.num_gens), ga_metric.avg_robots_reaching_goal)
    reached_goal_ax.set_ylabel('Average Number of Robots Reaching Goal')
    reached_goal_ax.set_xlabel('Generation')
    reached_goal_ax.set_title('Average Number of Robots Reaching Goal over generation')

    reached_goal_fig.savefig('reached_goal_fig.png', bbox_inches='tight')

    genome_dist_pd = ga_metric.get_genome_df()
    pal = sns.color_palette("colorblind")

    goal_gain_fig, goal_gain_ax = plt.subplots()
    sns.boxplot(x="Generation", y="Goal Gain", data=genome_dist_pd, palette=pal, ax=goal_gain_ax)
    goal_gain_ax.set_ylabel('Gain of Attractive Force Towards Goal')
    goal_gain_ax.set_xlabel('Generation')
    goal_gain_ax.set_title('Distribution of Population Goal Gain over generation')
    goal_gain_ax.set_xticks(np.arange(0, ga_config.num_gens, 2.0))

    goal_gain_fig.savefig('goal_gain_fig.png', bbox_inches='tight')

    obstacle_gain_fig, obstacle_gain_ax = plt.subplots()
    sns.boxplot(x="Generation", y="Obstacle Gain", data=genome_dist_pd, palette=pal, ax=obstacle_gain_ax)
    obstacle_gain_ax.set_ylabel('Gain of Repulsive Force From Obstacle')
    obstacle_gain_ax.set_xlabel('Generation')
    obstacle_gain_ax.set_title('Distribution of Population Obstacle Gain over generation')
    obstacle_gain_ax.set_xticks(np.arange(0, ga_config.num_gens, 2.0))

    obstacle_gain_fig.savefig('obstacle_gain_fig.png', bbox_inches='tight')

    obstacle_influence_fig, obstacle_influence_ax = plt.subplots()
    sns.boxplot(x="Generation", y="Obstacle Influence", data=genome_dist_pd, palette=pal, ax=obstacle_influence_ax)
    obstacle_influence_ax.set_ylabel('Radius of Obstacle Influence')
    obstacle_influence_ax.set_xlabel('Generation')
    obstacle_influence_ax.set_title('Distribution of Population Obstacle Influence over generation')
    obstacle_influence_ax.set_xticks(np.arange(0, ga_config.num_gens, 2.0))

    obstacle_influence_fig.savefig('obstacle_influence_fig.png', bbox_inches='tight')

    objective_val_fig, objective_val_ax = plt.subplots()
    sns.boxplot(x="Generation", y="Objective Value", data=genome_dist_pd, palette=pal, ax=objective_val_ax)
    objective_val_ax.set_ylabel('Objective Value')
    objective_val_ax.set_xlabel('Generation')
    objective_val_ax.set_title('Distribution of Population Objective Value over generation')
    objective_val_ax.set_xticks(np.arange(0, ga_config.num_gens, 2.0))

    obstacle_influence_fig.savefig('obstacle_influence_fig.png', bbox_inches='tight')