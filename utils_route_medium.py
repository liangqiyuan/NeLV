import json
import pandas as pd
import numpy as np
from haversine import haversine
import itertools
import re
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point, LineString
from matplotlib.colors import ListedColormap


class MediumRoutePlanner:
    def __init__(self):
        self.search_type_mapping = {"Pharmacy": 0, "Grocery": 1}
        self.stay_time_mapping = {0: 30, 1: 60}
        self.average_speed = 0.5 # km/min
        self.distance_factor = 1.5
        self.current_day = "Monday"
        self.departure_time = 10

        self.airports_df = pd.read_csv('./data/indiana_airports.csv')
        self.airports_df = self.airports_df.dropna(subset=['ICAO_ID'])

        self.system_prompt = """
Extract POIs, constraints, and weights from human instruction through step-by-step reasoning:

1) **Understand the conversation flow**
   - Identify whether the user is initiating a new route planning request or modifying an existing route.
   - If modifying, extract the specific change mentioned and update only that field in the existing JSON.

2) **Identify key locations**
   - Extract the start point and end point from the user's instruction.

3) **POIs Analysis**
    - Look for POIs mentioned that need to be visited.
    - Create list of unique POIs.

4) **Ensure consistency and correctness**
   - Always return a valid JSON object without any additional text, explanation, or error messages.

Output must be valid JSON with structure:
{
    "start_point": string,
    "end_point": string,
    "pois": ["poi1", "poi2", ...],
}
"""
        self.default_flight = {"start_point": "Indianapolis", "end_point": "Purdue University", "pois": ["Pharmacy", "Grocery"]}

    def find_closest_airport(self, name):
        return self.airports_df['name'].iloc[
            self.airports_df['name'].str.lower().map(
                lambda x: len(set(x.split()) & set(name.lower().split()))
            ).argmax()
        ]

    def find_mission_location(self, mission):
        data = []
        with open('data/indiana_yelp.json', 'r') as file:
            for line in file:
                record = json.loads(line)
                if record['categories'] and mission in record['categories']:
                    if record['hours'] and self.current_day in record['hours'] and record['hours'][self.current_day] != '0:0-0:0' and record['hours'][self.current_day] != '':
                        data.append({
                            'name': record['name'],
                            'latitude': record['latitude'],
                            'longitude': record['longitude'],
                            'stars': record['stars'],
                            'review_count': record['review_count'],
                            'hours': record['hours'][self.current_day],
                            'mission': mission,
                            'group': self.search_type_mapping[mission]
                        })
        return data

    def build_combined_graph(self, mission_data):
        mission_coords = [(loc['latitude'], loc['longitude']) for loc in mission_data]
        positions = {i: (coords[1], coords[0]) for i, coords in enumerate(mission_coords)}
        
        edge_indices = []
        edge_weights = []
        
        for i in range(len(mission_coords)):
            for j in range(len(mission_coords)):
                if i != j:
                    edge_indices.append([i, j])
                    edge_weights.append(haversine(mission_coords[i], mission_coords[j]))
        
        graph = {
            'positions': positions,
            'edge_indices': edge_indices,
            'edge_weights': edge_weights,
            'start_node': 0,
            'goal_node': len(mission_coords) - 1,
            'groups': [-1] + [loc['group'] for loc in mission_data[1:-1]] + [-1],
            'ratings': [-1.0] + [loc['stars'] for loc in mission_data[1:-1]] + [-1.0],
            'num_ratings': [-1] + [loc['review_count'] for loc in mission_data[1:-1]] + [-1],
            'openings': [''] + [str(loc['hours']) for loc in mission_data[1:-1]] + [''],
        }
        
        return graph

    def compute_path_and_cost(self, graph, selected_groups, node_in_groups):
        start_node = graph['start_node']
        goal_node = graph['goal_node']
        positions = graph['positions']
        groups = graph['groups']
        ratings = graph['ratings']
        num_ratings = graph['num_ratings']
        openings = graph['openings']
        alpha = 0.25
        beta = 0.5

        rating_range = (1, 5)
        travel_times = [haversine(positions[n1], positions[n2]) * self.distance_factor / self.average_speed / 60 for group in node_in_groups for n1 in group for n2 in group if n1 != n2]

        num_ratings_list = [num_ratings[node] for group in node_in_groups for node in group]
        travel_time_range = (min(travel_times), max(travel_times))
        num_ratings_range = (min(num_ratings_list), max(num_ratings_list))
        
        def min_max_normalize(value, value_range, zero_case=0.0):
            if value_range[0] == value_range[1]:
                return zero_case
            return (value - value_range[0]) / (value_range[1] - value_range[0])

        def calculate_real_time(node1, node2):
            travel_time = haversine(positions[node1], positions[node2]) * self.distance_factor / self.average_speed / 60
            stay_time = self.stay_time_mapping[groups[node2]] / 60 if groups[node2] != -1 else 0
            return travel_time + stay_time

        def calculate_weight(node1, node2):
            travel_time = min_max_normalize(haversine(positions[node1], positions[node2]) * self.distance_factor / self.average_speed / 60, travel_time_range, zero_case=0.0)
            if node2 == goal_node:
                return travel_time
            node_rating = min_max_normalize(ratings[node2], rating_range, zero_case=1.0)
            node_num_ratings = min_max_normalize(num_ratings[node2], num_ratings_range, zero_case=1.0)
            return - (alpha * node_rating + alpha * node_num_ratings) + beta * travel_time
        
        edges = [(start_node, node, calculate_weight(start_node, node)) 
                for node in node_in_groups[selected_groups[0]]]
        edges.extend([(prev_node, next_node, calculate_weight(prev_node, next_node))
                    for prev_group, next_group in zip(selected_groups[:-1], selected_groups[1:])
                    for prev_node in node_in_groups[prev_group]
                    for next_node in node_in_groups[next_group]])
        edges.extend([(node, goal_node, calculate_weight(node, goal_node)) 
                    for node in node_in_groups[selected_groups[-1]]])
        
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)
        _, path = nx.single_source_bellman_ford(G, source=start_node, target=goal_node, weight='weight')
        if not self.check_path_availability(path, groups, goal_node, positions, openings):
            return float('inf'), path

        real_time = self.departure_time + sum(calculate_real_time(path[i], path[i+1]) for i in range(len(path)-1))
        return real_time, path

    def MSGS(self, graph):
        start_node = graph['start_node']
        goal_node = graph['goal_node']
        groups = graph['groups']
        node_in_groups = []

        group_indices = np.unique([group_id for group_id in groups if group_id >= 0]).tolist()
        for group_id in group_indices:
            node_indices = np.where(np.array(groups) == group_id)[0]
            node_in_group = [idx for idx in node_indices if idx not in [start_node, goal_node]]
            node_in_groups.append(node_in_group)
        group_indices = list(range(len(node_in_groups)))

        best_weight = float('inf')
        best_path = None
        for num_groups in range(len(group_indices), 0, -1):
            for groups_subset in itertools.combinations(group_indices, num_groups):
                for perm in itertools.permutations(groups_subset):
                    w, p = self.compute_path_and_cost(graph, perm, node_in_groups)
                    if w < best_weight:
                        best_weight = w
                        best_path = p
                        route = best_path
                        break
                if best_path is not None:
                    break
            if best_path is not None:
                break     
        if best_path is None:
            route = [start_node, goal_node]

        return route

    def parse_time(self, time_str):
        times = time_str.split('-')
        converted_times = []
        
        for time in times:
            hour, minute = time.split(':')
            converted_time = float(hour) + float(minute)/60
            converted_times.append(converted_time)
        
        return converted_times

    def check_path_availability(self, path, groups, goal_node, positions, opening_hours):
        current_time = self.departure_time
        for i in range(len(path)-1):
            node1, node2 = path[i], path[i+1]
            travel_time = haversine(positions[node1], positions[node2]) * self.distance_factor / self.average_speed / 60
            current_time += travel_time
            if node2 != goal_node:
                day_opening_hours = opening_hours[node2]

                self.departure_time, end_time = self.parse_time(day_opening_hours)
                if not (self.departure_time <= current_time <= end_time):
                    return False
                current_time += self.stay_time_mapping[groups[node2]] / 60 if groups[node2] != -1 else 0
        return True


    def plot_routes(self, mission_data, route, start_idx, end_idx):
        fig, ax = plt.subplots(figsize=(10, 10))
        mission_colors = ['blue', 'orange', 'yellow', 'brown', 'pink', 'cyan', 'magenta']

        mission_df = pd.DataFrame(mission_data)
        geometry = [Point(lon, lat) for lon, lat in zip(mission_df['longitude'], mission_df['latitude'])]
        mission_gdf = gpd.GeoDataFrame(mission_df, geometry=geometry, crs="EPSG:4326")

        x_min, y_min, x_max, y_max = mission_gdf.geometry.total_bounds
        center_x, center_y = (x_max + x_min) / 2, (y_max + y_min) / 2
        half_size = max(x_max - x_min, y_max - y_min) / 2 * 1.2
        self.xylims = [[center_x - half_size, center_x + half_size], [center_y - half_size*609/790, center_y + half_size*609/790]]
        ax.set_xlim(self.xylims[0][0], self.xylims[0][1])
        ax.set_ylim(self.xylims[1][0], self.xylims[1][1])

        mission_idx = [x for x in route[1:-1]]
        mission_point = mission_gdf.iloc[mission_idx]
        unique_missions = mission_point['mission'].unique()
        mission_gdf.plot(ax=ax, column='mission', categorical=True, markersize=50, alpha=0.8, edgecolor='black', linewidth=2, cmap=ListedColormap(mission_colors[:len(unique_missions)]))

        mission_gdf.iloc[[0]].plot(ax=ax, color='red', marker='*', markersize=400, edgecolor='black', linewidth=2, label='Start Point', zorder=2)
        mission_gdf.iloc[[-1]].plot(ax=ax, color='purple', marker='D', markersize=100, edgecolor='black', linewidth=2, label='End Point', zorder=2)

        for i, mission_type in enumerate(unique_missions):
            subset = mission_point[mission_point['mission'] == mission_type]
            subset.plot(ax=ax, markersize=150, edgecolor='black', linewidth=2, color=mission_colors[i], label=mission_type, zorder=2)
    
        route_coords = []
        for idx in route:
            route_coords.append(mission_gdf.geometry.iloc[idx])
        route_line = gpd.GeoSeries([LineString(route_coords)], crs=mission_gdf.crs)
        route_line.plot(ax=ax, color='red', linewidth=2, label='Route', zorder=1)

        total_distance = 0
        for i in range(len(route)-1):
            from_idx = route[i]
            to_idx = route[i+1]
            from_point = mission_gdf.geometry.iloc[from_idx]
            to_point = mission_gdf.geometry.iloc[to_idx]
            dx = to_point.x - from_point.x
            dy = to_point.y - from_point.y
            distance = (dx**2 + dy**2)**0.5
            unit_dx = dx / distance
            unit_dy = dy / distance
            arrow_x = from_point.x + dx * 0.5
            arrow_y = from_point.y + dy * 0.5
            plt.annotate('', xy=(arrow_x + unit_dx*0.03, arrow_y + unit_dy*0.03), xytext=(arrow_x, arrow_y), arrowprops=dict(arrowstyle='wedge,tail_width=0.25', fc='red', ec='black', lw=2, mutation_scale=30), zorder=4)
            total_distance += haversine((mission_data[from_idx]['latitude'], mission_data[from_idx]['longitude']), (mission_data[to_idx]['latitude'], mission_data[to_idx]['longitude']))

        print(f"Total Distance: {total_distance:.2f} km")
        self.map_source = ctx.providers.OpenStreetMap.Mapnik
        ctx.add_basemap(ax, crs=mission_gdf.crs, source=self.map_source)
        ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=22, fancybox=True, shadow=True)
        plt.axis('off')
        plt.savefig('./temp/fig_route.png', bbox_inches='tight', pad_inches=0, dpi=150)

    def save_route_to_txt(self, mission_data, route):
        all_points = []
        for idx in route:
            point = mission_data[idx]
            all_points.append([point['latitude'], point['longitude'], 0])
        np.savetxt(f"./temp/route_coordinates.txt", all_points)

    def plan_route(self, planned_flight):
        for key in self.default_flight:
            if key not in planned_flight:
                planned_flight[key] = self.default_flight[key]

        start = self.find_closest_airport(planned_flight['start_point'])
        end = self.find_closest_airport(planned_flight['end_point'])
        start_idx = self.airports_df[self.airports_df['name'] == start].index[0]
        end_idx = self.airports_df[self.airports_df['name'] == end].index[0]
        start_airport = self.airports_df.iloc[start_idx]
        end_airport = self.airports_df.iloc[end_idx]

        mission_data = []
        for mission in planned_flight['pois']:
            mission_locations = self.find_mission_location(mission)
            mission_data.extend(mission_locations)
        
        mission_data = [{'latitude': start_airport['latitude'], 'longitude':start_airport['longitude']}] + mission_data + [{'latitude': end_airport['latitude'], 'longitude':end_airport['longitude']}]
        graph = self.build_combined_graph(mission_data)
        route = self.MSGS(graph)
        print(route)
        self.plot_routes(mission_data, route, start_idx, end_idx)
        self.save_route_to_txt(mission_data, route)

    