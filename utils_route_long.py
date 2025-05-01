import pandas as pd
import geopandas as gpd
import numpy as np
from haversine import haversine, Unit
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import LineString

class LongRoutePlanner:
    def __init__(self):
        self.tank_capacity = 80 # L
        self.miles_per_liter = 10

        self.airports_df = pd.read_csv('./data/usa_airports_fuel.csv')
        self.airports_df = self.airports_df.dropna(subset=['ICAO_ID'])
        self.airports_df = self.airports_df.dropna(subset=['state'])
        self.airports_df = self.airports_df[self.airports_df['state'] != 'HI']
        self.airports_df = self.airports_df[self.airports_df['state'] != 'PR']
        self.airports_df = self.airports_df[self.airports_df['state'] != 'VI']
        self.airports_df = self.airports_df[self.airports_df['state'] != 'AK']
        self.airports_df['fuel_price'] = self.airports_df['fuel_price'] / 3.78541
        self.airports_df = self.airports_df.reset_index(drop=True)

        self.system_prompt = """
Extract the task description and selection criteria from human instruction through step-by-step reasoning:

1) **Understand the conversation flow**
   - Identify whether the user is initiating a new route planning request or modifying an existing route.
   - If modifying, extract the specific change mentioned and update only that field in the existing JSON.

2) **Identify key locations**
   - Extract the start point and end point from the user's instruction.

3) **Determine the flight option**
   - The flight statoptionion must always be 'balanced', 'cheapest', or 'shortest'.

4) **Ensure consistency and correctness**
   - Always return a valid JSON object without any additional text, explanation, or error messages.

Output must be valid JSON with structure:
{
   "start_point": string,
   "end_point": string,
   "flight_option": string
}
"""
        self.default_flight = {"start_point": "New York", "end_point": "Los Angeles", "flight_option": "cheapest"}

    def find_closest_airport(self, name, airports_df):
        return airports_df['name'].iloc[
            airports_df['name'].str.lower().map(
                lambda x: len(set(x.split()) & set(name.lower().split()))
            ).argmax()
        ]

    def find_optimal_routes(self, airports_df, start_idx, end_idx, tank_capacity, miles_per_liter):
        range_capacity = tank_capacity * miles_per_liter
        G_distance = nx.DiGraph()
        G_cost = nx.DiGraph()
        G_balanced = nx.DiGraph()
        all_airports = set([start_idx, end_idx]) | set(airports_df[airports_df['fuel_price'].notna()].index)
        
        for airport_idx in all_airports:
            G_distance.add_node(airport_idx)
            G_cost.add_node(airport_idx)
            G_balanced.add_node(airport_idx)
        
        edges_data = []
        for to_idx in all_airports - {start_idx}:
            dist = haversine(
                (airports_df.loc[start_idx, 'latitude'], airports_df.loc[start_idx, 'longitude']),
                (airports_df.loc[to_idx, 'latitude'], airports_df.loc[to_idx, 'longitude']),
                unit=Unit.MILES
            )
            
            if dist <= range_capacity:
                edges_data.append({
                    'from': start_idx, 
                    'to': to_idx, 
                    'distance': dist, 
                })
        
        fuel_airports = airports_df[airports_df['fuel_price'].notna()].index
        for from_idx in fuel_airports:
            for to_idx in all_airports - {from_idx}:
                dist = haversine(
                    (airports_df.loc[from_idx, 'latitude'], airports_df.loc[from_idx, 'longitude']),
                    (airports_df.loc[to_idx, 'latitude'], airports_df.loc[to_idx, 'longitude']),
                    unit=Unit.MILES
                )
                
                if dist <= range_capacity:
                    to_fuel_price = airports_df.loc[to_idx, 'fuel_price']
                    if pd.notna(to_fuel_price):
                        refuel_cost = (dist / miles_per_liter + 5) * to_fuel_price
                    else:
                        refuel_cost = float('inf')
                    edges_data.append({
                        'from': from_idx, 
                        'to': to_idx, 
                        'distance': dist, 
                        'refuel_cost': refuel_cost,
                    })
        
        for edge in edges_data:
            G_distance.add_edge(edge['from'], edge['to'], weight=edge['distance'])
            G_cost.add_edge(edge['from'], edge['to'], weight=edge.get('refuel_cost', 0))
        
        if edges_data:
            distances = [edge['distance'] for edge in edges_data]
            costs = [edge.get('refuel_cost', 0) for edge in edges_data]
            min_dist, max_dist = min(distances), max(distances)
            min_cost, max_cost = min(costs), max(costs)
            for edge in edges_data:
                dist = edge['distance']
                cost = edge.get('refuel_cost', 0)
                norm_dist = (dist - min_dist) / (max_dist - min_dist)
                norm_cost = (cost - min_cost) / (max_cost - min_cost)
                G_balanced.add_edge(edge['from'], edge['to'], weight=0.5 * norm_dist + 0.5 * norm_cost)

        shortest_path = nx.dijkstra_path(G_distance, start_idx, end_idx, weight='weight')
        cheapest_path = nx.dijkstra_path(G_cost, start_idx, end_idx, weight='weight')
        balanced_path = nx.dijkstra_path(G_balanced, start_idx, end_idx, weight='weight')
        
        return {
            'shortest': shortest_path,
            'cheapest': cheapest_path,
            'balanced': balanced_path
        }

    def plot_routes(self, airports_df, routes, start_idx, end_idx):
        fig, ax = plt.subplots(figsize=(8, 8))
        airports_gdf = gpd.GeoDataFrame(airports_df, geometry=gpd.points_from_xy(airports_df['longitude'], airports_df['latitude']), crs="EPSG:4326")
        
        x_min, y_min, x_max, y_max = airports_gdf.geometry.total_bounds
        center_x, center_y = (x_max + x_min) / 2, (y_max + y_min) / 2
        half_size = max(x_max - x_min, y_max - y_min) / 2
        self.xylims = [[center_x - half_size, center_x + half_size], [center_y-7 - half_size*658/790, center_y-7 + half_size*658/790]]
        ax.set_xlim(self.xylims[0][0], self.xylims[0][1])
        ax.set_ylim(self.xylims[1][0], self.xylims[1][1])
        
        fuel_airports = airports_gdf[airports_df['fuel_price'].notna()]
        fuel_airports.plot(ax=ax, color='grey', markersize=10, alpha=0.5, label='Potential Fuel Stops')
        
        for route_type, route, color in [
            ('shortest', routes['shortest'], 'blue'),
            ('cheapest', routes['cheapest'], 'green'),
            ('balanced', routes['balanced'], 'yellow')
        ]:
            route_coords = airports_gdf.iloc[route].geometry
            route_line = gpd.GeoSeries([LineString(route_coords)], crs=airports_gdf.crs)
            route_line.plot(ax=ax, color=color, linewidth=2)
            fuel_idx = route[1:-1]
            fuel_point = airports_gdf.iloc[fuel_idx]
            fuel_point.plot(ax=ax, color=color, markersize=150, edgecolor='black', label=f'Fuel Stop ({route_type})', zorder=2)
            for i in range(len(route)-1):
                from_idx = route[i]
                to_idx = route[i+1]
                from_point = airports_gdf.geometry.iloc[from_idx]
                to_point = airports_gdf.geometry.iloc[to_idx]
                dx = to_point.x - from_point.x
                dy = to_point.y - from_point.y
                distance = (dx**2 + dy**2)**0.5
                if distance > 2:
                    unit_dx = dx / distance
                    unit_dy = dy / distance
                    arrow_x = from_point.x + dx * 0.5
                    arrow_y = from_point.y + dy * 0.5
                    # plt.arrow(arrow_x, arrow_y, unit_dx*0.01, unit_dy*0.01, head_width=0.5, head_length=0.8, fc=color, ec='black', zorder=4)
                    plt.annotate('', xy=(arrow_x + unit_dx*1.4, arrow_y + unit_dy*1.4), xytext=(arrow_x, arrow_y), arrowprops=dict(arrowstyle='wedge,tail_width=0.25', fc=color, ec='black', lw=1.5, mutation_scale=30), zorder=4)
    
        start_point = airports_gdf.iloc[[start_idx]]
        start_point.plot(ax=ax, color='red', marker='*', markersize=200, edgecolor='black', label='Start', zorder=3)
        end_point = airports_gdf.iloc[[end_idx]]
        end_point.plot(ax=ax, color='purple', marker='D', markersize=100, edgecolor='black', label='End', zorder=3)
        self.map_source = ctx.providers.Esri.WorldPhysical
        ctx.add_basemap(ax, crs=airports_gdf.crs, source=self.map_source)
        ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=16, fancybox=True, shadow=True)
        plt.axis('off')
        plt.savefig('./temp/fig_route.png', bbox_inches='tight', pad_inches=0)
    
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

        start = self.find_closest_airport(planned_flight['start_point'], self.airports_df)
        end = self.find_closest_airport(planned_flight['end_point'], self.airports_df)
        start_idx = self.airports_df[self.airports_df['name'] == start].index[0]
        end_idx = self.airports_df[self.airports_df['name'] == end].index[0]
        
        # routes = find_optimal_routes(airports_df, start_idx, end_idx, tank_capacity, miles_per_liter)
        routes = {'shortest': [1493, 732, 735, 296, 189], 'cheapest': [1493, 2179, 604, 1363, 1345, 117, 131, 216, 162, 176, 189], 'balanced': [1493, 1504, 741, 1132, 790, 789, 756, 117, 131, 216, 245, 162, 172, 163, 176, 189]}
        print(routes)
        
        self.plot_routes(self.airports_df, routes, start_idx, end_idx)
        self.save_route_to_txt(self.airports_df.to_dict('records'), routes[planned_flight['flight_option']])