# Return avg collisions, time steps, distance travelled
import pandas as pd
from collections import defaultdict

class GARobotMetrics:
    def __init__(self):
        self.avg_collisions = list()
        self.avg_time_steps = list()
        self.avg_distance_travelled = list()
        self.avg_robots_reaching_goal = list()
        
        self.genome_dist_dict = defaultdict(list)
        self.genome_dist_dict['Generation'] = list()
        self.genome_dist_dict['Goal Gain'] = list()
        self.genome_dist_dict['Obstacle Gain'] = list()
        self.genome_dist_dict['Obstacle Influence'] = list()
    
    def add_genome_dist(self, robots, no_generation):

        for i in range(len(robots)):
            robot = robots[i]
            self.genome_dist_dict['Generation'].append(no_generation)
            self.genome_dist_dict['Goal Gain'].append(robot.genome.to_goal_cost_gain)
            self.genome_dist_dict['Obstacle Gain'].append(robot.genome.obstacle_cost_gain)
            self.genome_dist_dict['Obstacle Influence'].append(robot.genome.obstacle_sphere_of_influence)
    
    def get_genome_df(self):
        genome_dist_pd = pd.DataFrame(self.genome_dist_dict)

        return genome_dist_pd
    
    def record_metrics(self, robots, reached_count, no_generation, runs_per_gen):
        collisions = [t.no_collisions for t in robots]
        time_steps = [t.time_steps for t in robots]
        distance_travelled = [t.distance_travelled for t in robots]
        
        denominator = (len(robots) * runs_per_gen)

        avg_collisions = sum(collisions) / denominator
        avg_time_steps = sum(time_steps) / denominator
        avg_distance_travelled = sum(distance_travelled) / denominator

        self.avg_collisions.append(avg_collisions)
        self.avg_time_steps.append(avg_time_steps)
        self.avg_distance_travelled.append(avg_distance_travelled)
        self.avg_robots_reaching_goal.append(reached_count/runs_per_gen)

        self.add_genome_dist(robots, no_generation)