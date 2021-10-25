"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
Modified by Megnath Ramesh for ECE 750 Project

"""

import math

import matplotlib.pyplot as plt
import numpy as np

show_animation = False


def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory

class EnvConfig:
    def __init__(self):
        self.env_range = [-1, 30]
        self.no_obstacles = 30
        self.ob_radius_range = [0,1]
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 1],
                            [5.0, 0],
                            [5.0, -1.5],
                            [7.0, -0.75],
                            [5.0, 0.8],
                            [8.0, 1.5],
                            [7.0, 1],
                            [8.0, -0.5],
                            [9.0, 0],
                            [12.0, 1],
                            [12.0, -0.25],
                            [15.0, 0.25],
                            [13.0, 0.5]
                            ])

class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # Parameters to tune for GARobot
        # Also used to check if goal is reached in both types
        self.to_goal_cost_gain = 25
        self.speed_cost_gain = 1
        self.obstacle_cost_gain = 5
        self.robot_radius = 0.25 # [m] for collision check

        # robot parameter
        self.max_speed = 1 # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 30 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 30 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 1.0  # [s]
        
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked


config = Config()
env_config = EnvConfig()

def motion(x, u, dt):
    """
    motion model
    """

    # x[2] += u[1] * dt
    x[0] += u[0] * dt
    x[1] += u[1] * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          config.min_speed, config.max_speed]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_accel * config.dt,
          x[4] + config.max_accel * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.v_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - np.sqrt(trajectory[-1, 3]**2 + trajectory[-1, 4]**2))
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                # if abs(best_u[0]) < config.robot_stuck_flag_cons \
                #         and abs(x[3]) < config.robot_stuck_flag_cons:
                #     # to ensure the robot do not get stuck in
                #     # best v=0 m/s (in front of an obstacle) and
                #     # best omega=0 rad/s (heading to the goal with
                #     # angle difference of 0)
                #     best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if np.array(r <= config.robot_radius).any():
        return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    # error_angle = math.atan2(dy, dx)
    # cost_angle = error_angle - trajectory[-1, 2]
    cost = np.sqrt(dx**2 + dy**2)
    return cost

def plot_robot(x, y, radius):  # pragma: no cover
    circle = plt.Circle((x, y), radius, color="b")
    plt.gcf().gca().add_artist(circle)


def main(sx=0.0, sy=0.0, gx=16.0, gy=0.0):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([sx, sy, 0, 0.0, 0.0])
    x_1 = np.array([sx+0.5, sy-0.5, 0, 0.0, 0.0])
    x_2 = np.array([sx+0.5, sy+0.5, 0, 0.0, 0.0])

    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]
    trajectory = np.array(x)
    trajectory_1 = np.array(x_1)
    trajectory_2 = np.array(x_2)
    ob = env_config.ob
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        u_1, predicted_trajectory_1 = dwa_control(x_1, config, goal, ob)
        x_1 = motion(x_1, u_1, config.dt)  # simulate robot
        trajectory_1 = np.vstack((trajectory_1, x_1))  # store state history

        u_2, predicted_trajectory_2 = dwa_control(x_2, config, goal, ob)
        x_2 = motion(x_2, u_2, config.dt)  # simulate robot
        trajectory_2 = np.vstack((trajectory_2, x_2))  # store state history

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")

            plt.plot(x[0], x[1], "xr")
            plot_robot(x[0], x[1], config.robot_radius)
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")

            plt.plot(x_1[0], x_1[1], "xr")
            plot_robot(x_1[0], x_1[1], config.robot_radius)
            plt.plot(predicted_trajectory_1[:, 0], predicted_trajectory_1[:, 1], "-g")

            plt.plot(x_2[0], x_2[1], "xr")
            plot_robot(x_2[0], x_2[1], config.robot_radius)
            plt.plot(predicted_trajectory_2[:, 0], predicted_trajectory_2[:, 1], "-g")

            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.01)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    
    plt.cla()
    plt.plot(goal[0], goal[1], "xb")
    plt.plot(ob[:, 0], ob[:, 1], "ok")
    plt.axis("equal")
    plt.grid(True)

    plt.plot(x[0], x[1], "xr")
    plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")

    plt.plot(x_1[0], x_1[1], "xr")
    plt.plot(trajectory_1[:, 0], trajectory_1[:, 1], "-r")

    plt.plot(x_2[0], x_2[1], "xr")
    plt.plot(trajectory_2[:, 0], trajectory_2[:, 1], "-r")

    plt.show()


if __name__ == '__main__':
    main()