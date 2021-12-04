'''
Main function for the GAROBOT project.

Author: Megnath Ramesh
'''

import numpy as np
import matplotlib.pyplot as plt

from genetic_alg import garobot, GARobotConfig, RobotType
from plot_utils import plot_final_results

show_animation = False

def main():

    # Define start point
    sx = 0
    sy = 5
    start = np.array([sx, sy])

    # Define goal point
    gx=10.0 
    gy=5.0
    goal = np.array([gx, gy])

    # Configure population size
    pop_size = 30

    # Create a configuration file for GAROBOT
    ga_config = GARobotConfig(50, 3, fixed=False, robot_type_gains=RobotType.FAST, clutter_pct=15)

    # Determine whether to show animation or not
    show_animation = False

    if(show_animation):
        # Create a plotting axis for animation
        _, anim_ax = plt.subplots(figsize=(8,8))
    else:
        anim_ax = None

    # Run the GAROBOT algorithm
    robots, env_config, ga_metric = garobot(pop_size, start, goal, ga_config, anim_ax, show_animation)

    # Plot and show final results
    plot_final_results(ga_metric, goal, env_config, ga_config)
    plt.show()

if __name__ == '__main__':
    main()