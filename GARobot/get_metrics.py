# Return avg collisions, time steps, distance travelled
def get_metrics(robots, runs_per_gen):
    collisions = [t.no_collisions for t in robots]
    time_steps = [t.time_steps for t in robots]
    distance_travelled = [t.distance_travelled for t in robots]
    
    denominator = (len(robots) * runs_per_gen)

    avg_collisions = sum(collisions) / denominator
    avg_time_steps = sum(time_steps) / denominator
    avg_distance_travelled = sum(distance_travelled) / denominator

    return (avg_collisions, avg_time_steps, avg_distance_travelled)