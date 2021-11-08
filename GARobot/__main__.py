import numpy as np
import matplotlib.pyplot as plt
from robot_factory import Robot

from motion_planner import run_generation

show_animation = False

def main():
    gx=30.0 
    gy=15.0
    
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])
    robots = []

    for i in range(5):
        robots.append(Robot.create_robot())

    run_generation(robots, goal)
    plt.show()


if __name__ == '__main__':
    main()