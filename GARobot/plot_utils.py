import matplotlib.pyplot as plt

def plot_robot(x, y, radius):  # pragma: no cover
    circle = plt.Circle((x, y), radius, color="b")
    plt.gcf().gca().add_artist(circle)

def plot_obstacles(obs):  # pragma: no cover
    for ob in obs:
        circle = plt.Circle((ob[0], ob[1]), ob[2], color="k")
        plt.gcf().gca().add_artist(circle)