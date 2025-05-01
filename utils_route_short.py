import pandas as pd
import numpy as np
import osmnx as ox
from haversine import haversine
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point, LineString
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


class ShortRoutePlanner:
    def __init__(self):
        self.airports_df = pd.read_csv('./data/indiana_airports.csv')
        self.airports_df = self.airports_df.dropna(subset=['ICAO_ID'])

        self.system_prompt = """
Extract POIs, constraints, and weights from human instruction through step-by-step reasoning:

1) **Understand the conversation flow**
   - Determine if this is a new route planning request or a modification to an existing route.
   - For modifications, identify only the specific changes requested and preserve existing parameters.

2) **Identify the key location**
   - Identify and extract the starting point from the user's instruction.

3) **Define target and operational range**
   - Extract the target name and operational range (in meters) from the user's instruction.

4) **Extract UAV count**
    - Count the total number of UAVs mentioned in the instruction.
    - If no specific count is provided, maintain previous value or default to 1.

5) **Ensure consistency and correctness**
   - Always return a valid JSON object without any additional text, explanation, or error messages.
   - Maintain all previously provided values unless explicitly changed by the user.

Output must be valid JSON with structure:
{
    "start_point": string,
    "target": string,
    "range": integer,
    "number_uavs": integer,
}
"""
        self.default_flight = {"start_point": "Purdue University", "target": "forests", "range": 5000, "number_uavs": 5}

    def find_closest_airport(self, name):
        return self.airports_df['name'].iloc[
            self.airports_df['name'].str.lower().map(
                lambda x: len(set(x.split()) & set(name.lower().split()))
            ).argmax()
        ]

    def get_mission_data(self, lat, lon, mission, radius):
        gdf = ox.features_from_point((lat, lon), mission, dist=radius)
        mission_data = []
        for idx, element in gdf.iterrows():
            centroid = element.geometry.centroid
            lat, lon = centroid.y, centroid.x
            mission_info = {
                'name': element.get('name', 'Unknown'),
                'latitude': lat,
                'longitude': lon
            }
            is_close_to_existing = False
            for existing_point in mission_data:
                point1 = (lat, lon)
                point2 = (existing_point['latitude'], existing_point['longitude'])
                distance = haversine(point1, point2)
                if distance <= 1:
                    is_close_to_existing = True
                    break
            if not is_close_to_existing:
                mission_data.append(mission_info)
        return mission_data

    def find_optimal_route(self, mission_data, num_agents):
        if num_agents == 1:
            G = nx.complete_graph(len(mission_data))
            for i in range(len(mission_data)):
                for j in range(i+1, len(mission_data)):
                    G.edges[i, j]['weight'] = haversine((mission_data[i]['latitude'], mission_data[i]['longitude']), (mission_data[j]['latitude'], mission_data[j]['longitude']))
            return [nx.approximation.traveling_salesman_problem(G, cycle=True)]
        
        data = {}
        distance_matrix = [[0 for _ in range(len(mission_data))] for _ in range(len(mission_data))]
        for i in range(len(mission_data)):
            for j in range(len(mission_data)):
                if i != j:
                    distance_matrix[i][j] = int(haversine((mission_data[i]['latitude'], mission_data[i]['longitude']), (mission_data[j]['latitude'], mission_data[j]['longitude'])) * 1000)

        
        data['distance_matrix'] = distance_matrix
        data['num_vehicles'] = num_agents
        data['depot'] = 0

        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        routing.AddDimension(transit_callback_index, 0, 30 * 1000, True, "Distance")
        distance_dimension = routing.GetDimensionOrDie("Distance")
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        solution = routing.SolveWithParameters(search_parameters)

        routes = []
        if solution:
            for vehicle_id in range(data['num_vehicles']):
                route = []
                index = routing.Start(vehicle_id)
                while not routing.IsEnd(index):
                    route.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index)) 
                routes.append(route)
        return routes

    def plot_routes(self, mission_data, routes):
        fig, ax = plt.subplots(figsize=(8, 8))
        mission_df = pd.DataFrame(mission_data)
        mission_df['type'] = ['Start' if i == 0 else 'Mission' for i in range(len(mission_data))]
        geometry = [Point(p['longitude'], p['latitude']) for p in mission_data]
        mission_gdf = gpd.GeoDataFrame(mission_df, geometry=geometry, crs="EPSG:4326")

        x_min, y_min, x_max, y_max = mission_gdf.geometry.total_bounds
        center_x, center_y = (x_max + x_min) / 2, (y_max + y_min) / 2
        half_size = max(x_max - x_min, y_max - y_min) / 2 * 1.1
        self.xylims = [[center_x - half_size, center_x + half_size], [center_y - half_size*606/790, center_y + half_size*606/790]]
        ax.set_xlim(self.xylims[0][0], self.xylims[0][1])
        ax.set_ylim(self.xylims[1][0], self.xylims[1][1])
        
        airport = mission_gdf[mission_gdf['type'] == 'Start']
        mission = mission_gdf[mission_gdf['type'] == 'Mission']
        airport.plot(ax=ax, color='red', marker='*', markersize=200, edgecolor='black', label='Start', zorder=3)
        mission.plot(ax=ax, color='green', marker='o', markersize=100, edgecolor='black', label='Mission Points', zorder=2)

        colors = ['blue', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'olive', 'navy', 'teal']
        total_distance = 0

        # ax.plot([], [], color='blue', linewidth=2, linestyle='-', label='UAV Route')
        for i, route in enumerate(routes):
            color = colors[i % len(colors)]
            route_points = [mission_gdf.geometry.iloc[j] for j in route]
            route_line = gpd.GeoSeries([LineString(route_points)], crs=mission_gdf.crs)
            route_line.plot(ax=ax, color=color, linewidth=2, linestyle='-', label=f'UAV Route', zorder=1)
            if route == [0, 0]:
                continue

            agent_distance = 0
            for j in range(len(route)-1):
                from_idx, to_idx = route[j], route[j+1]
                from_point, to_point = mission_gdf.geometry.iloc[from_idx], mission_gdf.geometry.iloc[to_idx]
                dx, dy = to_point.x - from_point.x, to_point.y - from_point.y
                distance = (dx**2 + dy**2)**0.5
                unit_dx, unit_dy = dx / distance, dy / distance
                arrow_x, arrow_y = from_point.x + dx * 0.5, from_point.y + dy * 0.5
                plt.annotate('', xy=(arrow_x + unit_dx*0.003, arrow_y + unit_dy*0.003), xytext=(arrow_x, arrow_y), 
                            arrowprops=dict(arrowstyle='wedge,tail_width=0.25', fc=color, ec='black', lw=1.5, mutation_scale=30), zorder=4)
                segment_distance = haversine((mission_data[from_idx]['latitude'], mission_data[from_idx]['longitude']), (mission_data[to_idx]['latitude'], mission_data[to_idx]['longitude']))
                agent_distance += segment_distance
            
            total_distance += agent_distance
            print(f"UAV {i+1} Distance: {agent_distance:.2f} km")
        

        legend_elements = [Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markeredgecolor='black', markersize=20, label='Start'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markeredgecolor='black', markersize=15, label='Mission Points')]
        
        class HandlerColorLine(HandlerBase):
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                num_stripes = len(orig_handle.get_colors())
                stripe_width = width / num_stripes
                segments = []
                for i, color in enumerate(orig_handle.get_colors()):
                    segment = Line2D([xdescent + i * stripe_width, xdescent + (i + 1) * stripe_width], [ydescent + height / 2, ydescent + height / 2], color=color, linewidth=orig_handle.get_linewidth())
                    segment.set_transform(trans)
                    segments.append(segment)
                return segments
        
        class ColorLine(Line2D):
            def __init__(self, colors, **kwargs):
                super().__init__([0], [0], **kwargs)
                self._colors = colors
            def get_colors(self):
                return self._colors
        
        rainbow_colors = colors[:len(routes)]
        colorline = ColorLine(rainbow_colors, linewidth=2)
        legend_elements.append(colorline)
        ax.legend(legend_elements, ['Start', 'Mission Points', 'UAV Route'], handler_map={ColorLine: HandlerColorLine()}, 
                loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=22, fancybox=True, shadow=True)


        print(f"Total Distance (all agents): {total_distance:.2f} km")
        self.map_source = ctx.providers.Esri.WorldImagery
        ctx.add_basemap(ax, crs=mission_gdf.crs, source=self.map_source)
        # ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=22, fancybox=True, shadow=True)
        plt.axis('off')
        plt.savefig('./temp/fig_route.png', bbox_inches='tight', pad_inches=0)
    
    def save_route_to_txt(self, mission_data, routes):
        all_points = []
        for route_id, route in enumerate(routes):
            for idx in route:
                point = mission_data[idx]
                all_points.append([point['latitude'], point['longitude'], 0])
            np.savetxt(f"./temp/route_coordinates_{route_id}.txt", all_points)

    def plan_route(self, planned_flight):
        for key in self.default_flight:
            if key not in planned_flight:
                planned_flight[key] = self.default_flight[key]

        print(planned_flight)
        mapping = {'forests': {'natural': 'wood'}}

        start = self.find_closest_airport(planned_flight['start_point'])
        start_idx = self.airports_df[self.airports_df['name'] == start].index[0]
        start_airport = self.airports_df.iloc[start_idx]
        mission_data = [{'name': 'airport', 'latitude': start_airport['latitude'], 'longitude': start_airport['longitude']}]
        mission_data.extend(self.get_mission_data(start_airport['latitude'], start_airport['longitude'], mapping[planned_flight['target']], planned_flight['range']))

        routes = self.find_optimal_route(mission_data, planned_flight['number_uavs'])
        print(routes)
        self.plot_routes(mission_data, routes)
        self.save_route_to_txt(mission_data, routes)


    