import matplotlib.pyplot as plt

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

    anim_ax.set_xlim(env_config.env_range)
    anim_ax.axis("equal")
    anim_ax.grid(True)
    plt.pause(0.0001)