import numpy as np
import matplotlib.pyplot as plt
from robot_factory import Robot

from genetic_alg import garobot, GARobotConfig

show_animation = False

def main():

    # Define goal point
    gx=10.0 
    gy=5.0

    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])
    pop_size = 30

    config = GARobotConfig()

    garobot(pop_size, goal, config)
    
    plt.show()


if __name__ == '__main__':
    main()