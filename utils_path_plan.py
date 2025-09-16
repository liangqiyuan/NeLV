import numpy as np
import pandas as pd
import copy
import os
import xarray as xr
import pymap3d as pm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.legend_handler import HandlerPatch
import geopandas as gpd
from shapely.geometry import LineString, box, Polygon, Point
import ast
from rtree import index
from pyproj import Transformer
import contextily as ctx
from herbie import Herbie


class Path_Planner():
    def __init__(self):
        self.rng = np.random.default_rng(2024)

    def plan_path(self, file_path, n_generation, xylims, map_source):

        # u_wind = xr.open_dataset(f"./data/UGRD.nc", decode_timedelta=False)
        # v_wind = xr.open_dataset(f"./data/VGRD.nc", decode_timedelta=False)
        year = '2025'; month = '02'; day = '15'
        H = Herbie(
            f'{year}-{month}-{day}',
            model="hrrr",
            product="prs",
            fxx=0,
        )
        sample_weather = H.xarray(f"VGRD:{'250 mb'}")

        with open(file_path, "r") as file:
            lines = file.readlines()

        path_latitude = []
        path_longitude = []
        for i in lines:
          path_latitude.append(list(map(float, i.split(' ')))[0])
          path_longitude.append(list(map(float, i.split(' ')))[1])

        # Based on mission's bound
        bound_north = [max(sample_weather['latitude'].values[0, 0], min(path_latitude) - 1),
                 min(sample_weather['latitude'].values[-1, 0] - 2.5, max(path_latitude) + 1)]
        bound_west = [max((sample_weather['longitude'].values[0, 0] + 180) % 360 - 180, min(path_longitude) - 1),
                min((sample_weather['longitude'].values[0, -1] + 180) % 360 - 180, max(path_longitude) + 1)]
        if abs(bound_north[0]-bound_north[1]) > 8 or abs(bound_west[0]-bound_west[1]) > 8:
            long_range = True
        else:
            long_range = False

        sfip, cape, brn, u_wind, v_wind, lons, lats = self.weather_data(year, month, day, long_range)

        w_0 = 10 # constraint penalties
        w_1 = 1 # path distance
        w_2 = 0 # fuel consumption
        w_3 = 0.05 # weather
        w_4 = 0.1 # ground risk

        N = 50
        n_points = 2
        cruise_h = 1000
        all_best_path = []
        left_bottom_corner = [u_wind['latitude'].values[0, 0], ((u_wind['longitude'].values[0, 0]+180)%360)-180]
        max_ground_risk = 10
        W, wind_direction = self.get_wind(u_wind['u'].values, v_wind['v'].values, new=True)

        for num_line in range(0, len(lines)-1):
            p_s = np.array(list(map(float, lines[num_line].split(' '))))
            p_e = np.array(list(map(float, lines[num_line+1].split(' '))))

            # Based on mission's bound
            small_bound_north = [max(sample_weather['latitude'].values[0, 0], min(path_latitude[num_line:(num_line+2)]) - 0.5),
                           min(sample_weather['latitude'].values[-1, 0] - 2.5, max(path_latitude[num_line:(num_line+2)]) + 0.5)]
            small_bound_west = [max((sample_weather['longitude'].values[0, 0] + 180) % 360 - 180, min(path_longitude[num_line:(num_line+2)]) - 0.5),
                          min((sample_weather['longitude'].values[0, -1] + 180) % 360 - 180, max(path_longitude[num_line:(num_line+2)]) + 0.5)]
            range_min = np.array([small_bound_north[0], small_bound_west[0], 0])
            range_max = np.array([small_bound_north[1], small_bound_west[1], cruise_h])

            x1, y1, z = pm.geodetic2enu(range_min[0], range_min[1], 0, left_bottom_corner[0], left_bottom_corner[1], 0)
            x2, y2, z = pm.geodetic2enu(range_max[0], range_max[1], 0, left_bottom_corner[0], left_bottom_corner[1], 0)
            bound_x_enu = np.arange(x1/1000, x2/1000, 3)
            bound_y_enu = np.arange(y1/1000, y2/1000, 3)

            airspace_geo = self.load_airspace(small_bound_north, small_bound_west, long_range)
            all_cities_geo, ground_level = self.load_cities(small_bound_west)

            spatial_index, airspace_geoms = self.create_idx(airspace_geo)
            city_spatial_index, city_geoms = self.create_idx(all_cities_geo)

            shortest_path = []
            x, y, z = pm.geodetic2enu(p_s[0], p_s[1], 0, left_bottom_corner[0], left_bottom_corner[1], 0)
            shortest_path.append(np.array([x/1000, y/1000]))
            x, y, z = pm.geodetic2enu(p_e[0], p_e[1], 0, left_bottom_corner[0], left_bottom_corner[1], 0)
            shortest_path.append(np.array([x/1000, y/1000]))
            shortest_path = np.stack(shortest_path)
            shortest_path_length = np.sum(np.sqrt(np.sum(np.square(shortest_path[1:, :] - shortest_path[0:-1]), axis=1)))

            smallest_time = 1
            max_weather_risk = 1e3

            temp = self.rng.random((N, n_points, 3))
            population = np.zeros((N, n_points+2, 3))
            velocity = self.rng.random((N, n_points+2, 3))
            if long_range:
              velocity_up = (range_max - range_min) * 0.4
            else:
              velocity_up = (range_max - range_min) * 0.1
            velocity_lo = -velocity_up

            all_x_penalties = []
            all_x_constraint = []

            for i in range(N):
                population[i] = np.vstack((p_s, (range_max - range_min) * temp[i] + range_min, p_e))
                velocity[i] = (velocity_up-velocity_lo)*velocity[i]+velocity_lo
            population[:, :, 2] = cruise_h

            p = population.copy()
            [constraint_violation, penalties, weather_penalties, ground_penalties] = (
                    self.get_fitness(population, spatial_index, airspace_geoms, W, wind_direction,
                                        sfip, cape, brn, city_spatial_index, city_geoms, ground_level, range_min, range_max, left_bottom_corner, bound_x_enu, bound_y_enu, long_range))
            initial_penalties = (w_0 * np.sum(constraint_violation, axis=1) + w_1 * penalties[:, 0] + w_2 * penalties[:, 1] + w_3 * weather_penalties + w_4 * ground_penalties)
            g = np.min(initial_penalties)
            best_path = population[np.argmin(initial_penalties), :]

            for i in range(n_generation+1):
                velocity *= -0.5/n_generation*i + 1
                new_population = self.update_population(population, velocity, p, best_path)

                new_population[:, :, 2] = cruise_h
                new_population[:, 0, :] = p_s
                new_population[:, -1, :] = p_e
                p[:, :, 2] = cruise_h
                p[:, 0, :] = p_s
                p[:, -1, :] = p_e

                [x_constraint_violation, x_penalties, x_weather_penalties, x_ground_penalties] = (
                    self.get_fitness(new_population, spatial_index, airspace_geoms, W, wind_direction,
                                        sfip, cape, brn, city_spatial_index, city_geoms, ground_level, range_min, range_max, left_bottom_corner, bound_x_enu, bound_y_enu, long_range))
                [p_constraint_violation, p_penalties, p_weather_penalties, p_ground_penalties] = (
                    self.get_fitness(p, spatial_index, airspace_geoms, W, wind_direction,
                                        sfip, cape, brn, city_spatial_index, city_geoms, ground_level, range_min, range_max, left_bottom_corner, bound_x_enu, bound_y_enu, long_range))

                x_real_penalties = (w_0 * np.sum(x_constraint_violation, axis=1) +
                                    w_1 * (1-shortest_path_length/x_penalties[:, 0]) +
                                    w_2 * (1-smallest_time/x_penalties[:, 1]) +
                                    w_3 * x_weather_penalties/max_weather_risk + w_4 * x_ground_penalties/max_ground_risk)

                all_x_penalties.append(x_real_penalties)
                all_x_constraint.append(np.sum(x_constraint_violation, axis=1))
                p_real_penalties = (w_0 * np.sum(p_constraint_violation, axis=1) +
                                    w_1 * (1-shortest_path_length/p_penalties[:, 0]) +
                                    w_2 * (1-smallest_time/p_penalties[:, 1]) +
                                    w_3 * p_weather_penalties/max_weather_risk + w_4 * p_ground_penalties/max_ground_risk)


                idx = np.where(x_real_penalties <= p_real_penalties)
                p[idx] = new_population[idx]
                p_real_penalties[idx] = x_real_penalties[idx]
                if np.any(p_real_penalties < g):
                    g = np.min(p_real_penalties)
                    best_path = p[np.argmin(p_real_penalties)]
                if np.any(x_penalties[:, 0] < shortest_path_length):
                    raise Exception("Shorter than the shortest path!")
                population = new_population.copy()

                # np.savetxt(f'./temp/path_coordinates_{num_line}_{i}.txt', best_path)

            all_best_path.append(best_path[0:-1])

        all_best_path.append(np.array(list(map(float, lines[-1].split(' ')))))
        all_best_path = np.vstack(all_best_path)
        best_path[0,2] = 0
        best_path[-1,2] = 0

        np.savetxt('./temp/path_coordinates.txt', all_best_path)
        land_mark = []
        for i in range(1, len(lines) - 1):
            land_mark.append(np.array(list(map(float, lines[i].split(' ')))))
        airspace_geo = self.load_airspace(bound_north, bound_west, long_range)
        print(len(airspace_geo))
        all_cities_geo, ground_level = self.load_cities(bound_west)
        self.plot_path(all_best_path, airspace_geo, all_cities_geo, sfip, brn, cape, lons, lats, land_mark, n_points, long_range, xylims, map_source)

    def weather_data(self, year, month, day, long_range):

        H = Herbie(
            f'{year}-{month}-{day}',
            model="hrrr",
            product="prs",
            fxx=0,
        )

        if long_range:
            altitude = '250 mb'
        else:
            altitude = '950 mb'
        clwmr = H.xarray(f"CLMR:{altitude}")  # Cloud Mixing Ratio [kg/kg]
        cice = H.xarray(f"CIMIXR:{altitude}")  # Cloud Ice Mixing Ratio [kg/kg]
        spfh = H.xarray(f"SPFH:{altitude}")  # Specific Humidity [kg/kg]
        rwmr = H.xarray(f"RWMR:{altitude}")  # Rain Mixing Ratio [kg/kg]
        snmr = H.xarray(f"SNMR:{altitude}")  # Snow Mixing Ratio [kg/kg]
        t_data = H.xarray(f"TMP:{altitude}")
        t = t_data['t'].values - 273.15  # Temperature [K]
        rh = (H.xarray(f"RH:{altitude}"))['r'].values  # Relative Humidity [%]
        vvel = (H.xarray(f"VVEL:{altitude}"))['w'].values  # Vertical Velocity (Pressure) [Pa/s]
        cape = (H.xarray("CAPE:255-0 mb above ground"))['cape'].values  # Vertical Velocity (Pressure) [Pa/s]
        vucsh = (H.xarray("VUCSH:0-6000 m above ground"))['vucsh'].values  # Vertical Velocity (Pressure) [Pa/s]
        vvcsh = (H.xarray("VVCSH:0-6000 m above ground"))['vvcsh'].values  # Vertical Velocity (Pressure) [Pa/s]
        ugrd = H.xarray(f"UGRD:{altitude}")  # Vertical Velocity (Pressure) [Pa/s]
        vgrd = H.xarray(f"VGRD:{altitude}")  # Vertical Velocity (Pressure) [Pa/s]
        
        # Get longitude and latitude coordinates
        lons = t_data.longitude.values
        lats = t_data.latitude.values

        m_rh = copy.copy(rh)
        m_t = copy.copy(t)
        m_vvel = copy.copy(vvel)
        m_cape = copy.copy(cape)

        brn = copy.copy(cape)/(0.5*(vucsh**2 + vvcsh**2))
        m_brn = copy.copy(brn)

        lwc = 1000 * clwmr['clwmr'].values / (
                    cice['unknown'].values + (spfh['q'].values/(1-spfh['q'].values)) + rwmr['rwmr'].values + snmr['snmr'].values)
        m_lwc = copy.copy(lwc)
        m_lwc[lwc <= 0.4] /= 0.4
        m_lwc[lwc > 0.4] = 1

        m_vvel[vvel <= -0.5] = 1
        m_vvel[(vvel > -0.5) & (vvel <= 0)] /= -0.5
        m_vvel[(vvel > 0) & (vvel <= 1)] *= -0.4
        m_vvel[vvel > 1] = -0.4

        m_rh[rh <= 60] = 0
        m_rh[(rh > 60) & (rh <= 97)] = ((rh[(rh > 60) & (rh <= 97)] - 60)/37)**2
        m_rh[rh > 97] = 1

        m_t[t < -28] = 0
        m_t[(t >= -28) & (t <= -12)] = (t[(t >= -28) & (t <= -12)] + 28)/16
        m_t[(t > -12) & (t <= 0)] = 1
        m_t[(t > 0) & (t <= 0.1)] = 1 - (t[(t > 0) & (t <= 0.1)])*10
        m_t[t > 1] = 0

        m_cape[cape <= 0] = 0
        m_cape[(cape > 0) & (cape <= 1000)] = 1
        m_cape[(cape > 1000) & (cape <= 2500)] = 2
        m_cape[(cape > 2500) & (cape <= 3500)] = 3
        m_cape[cape > 3500] = 4

        m_brn[brn <= 10] = 0
        m_brn[(brn > 10) & (brn <= 50)] = 1
        m_brn[brn > 50] = 2

        alpha = 0.35
        beta = 0.2
        gamma = 0.45
        sfip = m_t * (alpha*m_rh + beta*m_vvel + gamma*m_lwc)
        sfip[sfip < 0] = 0

        return sfip, m_cape, m_brn, ugrd, vgrd, lons, lats
    
    def create_idx(self, object):
        spatial_index = index.Index()
        airspace_geoms = []
        for idx, area in enumerate(object):
            airspace_geom = Polygon(area)
            airspace_geoms.append(airspace_geom)
            spatial_index.insert(idx, airspace_geom.bounds)
        return spatial_index, airspace_geoms

    def load_cities(self, bound_west):
        df = pd.read_csv('./data/indiana_cities_with_area.csv')
        # x = df['Bounding Box']
        all_bbox = []
        ground_level = []

        for i, item in enumerate(df['Bounding Box'].apply(ast.literal_eval)):
            temp = []
            temp.append([item[0], item[2]])
            temp.append([item[0], item[3]])
            temp.append([item[1], item[3]])
            temp.append([item[1], item[2]])
            temp.append([item[0], item[2]])
            all_bbox.append(temp)
            if df['Population Density'][i] >= 500:
                ground_level.append(10)
            elif df['Population Density'][i] >= 100:
                ground_level.append(5)
            elif df['Population Density'][i] >= 10:
                ground_level.append(1)
            else:
                ground_level.append(0)


        box_idx = len(all_bbox) - 1
        while box_idx >= 0:
            if np.max(np.stack(all_bbox[box_idx])[:, 1]) > bound_west[1]:
                all_bbox.pop(box_idx)
            box_idx -= 1

        return all_bbox, np.stack(ground_level)
    
    def prep_loc(self, left_bottom_corner, all_cities, p_s, p_e, land_mark, airspaces, bound_north, bound_west):
        # Convert geodetic to ENU (East-North-Up)
        # width = 1799 * 3
        # height = 1059 * 3
        range_max = []
        range_min = []

        new_cities = []
        for city in all_cities:
            geo_loc = np.stack(city)
            x, y, z = pm.geodetic2enu(left_bottom_corner[0], left_bottom_corner[1], 0, geo_loc[:, 0], geo_loc[:, 1], 0)
            distance_loc = (np.stack((np.abs(x) / 1000, np.abs(y) / 1000, np.zeros((geo_loc.shape[0],))))).T
            new_cities.append(distance_loc)

        x, y, z = pm.geodetic2enu(p_s[0], p_s[1], 0, left_bottom_corner[0], left_bottom_corner[1], 0)
        p_s[0] = np.abs(x) / 1000
        p_s[1] = np.abs(y) / 1000
        x, y, z = pm.geodetic2enu(p_e[0], p_e[1], 0, left_bottom_corner[0], left_bottom_corner[1], 0)
        p_e[0] = np.abs(x) / 1000
        p_e[1] = np.abs(y) / 1000
        x, y, z = pm.geodetic2enu(land_mark[0], land_mark[1], 0, left_bottom_corner[0], left_bottom_corner[1], 0)
        land_mark[0] = np.abs(x) / 1000
        land_mark[1] = np.abs(y) / 1000

        x, y, z = pm.geodetic2enu(bound_north[0], bound_west[0], 0, left_bottom_corner[0], left_bottom_corner[1], 0)
        range_min.append(np.abs(x) / 1000)
        range_min.append(np.abs(y) / 1000)
        range_min.append(0)

        x, y, z = pm.geodetic2enu(bound_north[1], bound_west[1], 0, left_bottom_corner[0], left_bottom_corner[1], 0, )
        range_max.append(np.abs(x) / 1000)
        range_max.append(np.abs(y) / 1000)
        range_max.append(5)

        new_airspace = []
        for airspace in airspaces:
            geo_loc = np.stack(airspace)
            x, y, z = pm.geodetic2enu(geo_loc[:, 0], geo_loc[:, 1], 0, left_bottom_corner[0], left_bottom_corner[1], 0, )
            distance_loc = (np.stack((np.abs(x) / 1000, np.abs(y) / 1000, np.zeros((geo_loc.shape[0],))))).T
            new_airspace.append(distance_loc)

        return land_mark, p_s, p_e, new_cities, new_airspace, np.stack(range_min), np.stack(range_max)

    def prep_loc_new(self, left_bottom_corner, p_s, p_e, land_mark, bound_north, bound_west):
        # Convert geodetic to ENU (East-North-Up)
        # width = 1799 * 3
        # height = 1059 * 3
        range_max = []
        range_min = []
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        x_0, y_0 = transformer.transform(left_bottom_corner[1], left_bottom_corner[0])

        x, y = transformer.transform(p_e[1], p_e[0])
        p_e[0] = np.abs(y-y_0) / 1000
        p_e[1] = np.abs(x-x_0) / 1000
        x, y = transformer.transform(p_s[1], p_s[0])
        p_s[0] = np.abs(y - y_0) / 1000
        p_s[1] = np.abs(x - x_0) / 1000

        x, y = transformer.transform(land_mark[1], land_mark[0])
        land_mark[0] = np.abs(y-y_0) / 1000
        land_mark[1] = np.abs(x-x_0) / 1000

        x, y = transformer.transform(bound_west[0], bound_north[0])
        range_min.append(np.abs(y-y_0) / 1000)
        range_min.append(np.abs(x-x_0) / 1000)
        range_min.append(0)

        x, y = transformer.transform(bound_west[1], bound_north[1])
        range_max.append(np.abs(y-y_0) / 1000)
        range_max.append(np.abs(x-x_0) / 1000)
        range_max.append(5)

        return land_mark, p_s, p_e, np.stack(range_min), np.stack(range_max)

    def load_airspace(self, bound_north, bound_west, long_range):
        file_path = './data/us_asp.txt'
        with open(file_path, "r") as file:
            lines = file.readlines()

        all_airspace = []
        temp = []
        record = False


        for i in range(len(lines) - 1):
            line = lines[i]
            split_line = line.split(' ')
            if line[0:2] == 'AN':
                if line[-6:-1] == 'NOTAM':
                    record = True
                else:
                    record = False
            if line[0:2] == 'AL':
                try:
                    alt = int(split_line[1][0:-2])
                except:
                    alt = 1e6
                if line[3:6] == 'GND':
                    alt = 0
                if alt <= 40000:  # 1500:
                    temp_split = lines[i + 1].split(' ')
                    if temp_split[1][0:2] == 'FL':
                        alt_up = int(temp_split[1][2:])*100
                    else:
                        alt_up = int(temp_split[1][0:-2])
                    if long_range:
                        if alt_up >= 15000:
                            record = True
                        else:
                            record = False
                    else:
                        # if line[3:6] == 'GND' or (0 <= alt <= 2000 and alt_up <= 2000):
                        if alt <= 2000 and alt_up > 0 and alt_up > alt:
                            record = True
                        else:
                            record = False
                else:
                    record = False
            if record:
                if line[0:2] == 'DP':
                    north = list(map(float, line[3:14].split(':')))
                    west = list(map(float, line[19:-3].split(':')))
                    temp.append([north[0] + north[1] / 60 + north[2] / 3600, -(west[0] + west[1] / 60 + west[2] / 3600)])
                    if not lines[i + 1].strip():
                        all_airspace.append(temp)
                        temp = []

        inbound_airspace = []
        for airspace in all_airspace:
            for points in airspace:
                if bound_north[0] < points[0] < bound_north[1]:
                    if bound_west[0] < points[1] < bound_west[1]:
                        inbound_airspace.append(airspace)
                        break
        airspace_idx = 0
        while airspace_idx < len(inbound_airspace):
            temp = np.stack(inbound_airspace[airspace_idx])
            if np.max(temp[:, 1]) > bound_west[1]:
                inbound_airspace.pop(airspace_idx)
            else:
                airspace_idx += 1
        for i, airspace in enumerate(inbound_airspace):
            if airspace[0] != airspace[-1]:
                inbound_airspace[i].append(airspace[0])

        return inbound_airspace

    def check_cylinders(self, p1, p2, cy_loc, cy_shape, ground_level, ground):
        temp = p2 - p1
        line_length = np.sqrt(np.sum(np.square(temp), axis=1))
        A = temp[:, 1] / temp[:, 0]
        C = p1[:, 1] - A * p1[:, 0]
        a = 1 + np.square(A)
        count = 0
        ground_penalties = 0

        for j in range(0, np.shape(cy_loc)[0]):

            b = 2 * A * C - 2 * A * cy_loc[j, 1] - 2 * cy_loc[j, 0]
            c = np.square(cy_loc[j, 0]) + np.square(cy_loc[j, 1]) + np.square(C) - 2 * cy_loc[j, 1] * C - np.square(
                cy_shape[j, 0])

            circle_check = np.square(p1[:, 0] - cy_loc[j, 0]) + np.square(p1[:, 1] - cy_loc[j, 1]) - np.square(
                cy_shape[j, 0])
            find_temp = np.where(circle_check[0:-1] < 0 & (p1[0:-1, 2] <= cy_shape[j, 1]))[0]
            if find_temp.size > 0:
                find_temp = np.concatenate((find_temp, find_temp - 1))

            check = b * b - 4 * a * c
            collide_line = [0]

            if np.any(check > 0):
                idx = np.where(check > 0)[0]

                a_idx = a[idx]
                b_idx = b[idx]
                c_idx = c[idx]

                int_1 = (-b_idx + np.sqrt(b_idx * b_idx - 4 * a_idx * c_idx)) / (2 * a_idx)
                int_2 = (-b_idx - np.sqrt(b_idx * b_idx - 4 * a_idx * c_idx)) / (2 * a_idx)

                t_1 = (int_1 - p1[idx, 0]) / temp[idx, 0]
                t_2 = (int_2 - p1[idx, 0]) / temp[idx, 0]

                z_1 = np.squeeze(p1[idx, 2] + temp[idx, 2] * t_1)
                z_2 = np.squeeze(p1[idx, 2] + temp[idx, 2] * t_2)


                all_int_1 = np.column_stack((int_1, A[idx] * int_1 + C[idx], z_1))
                all_int_2 = np.column_stack((int_2, A[idx] * int_2 + C[idx], z_2))


                dist_1 = np.max(np.array([np.sqrt(np.sum(np.square(all_int_1 - p1[idx]), axis=1)),
                                          np.sqrt(np.sum(np.square(all_int_1 - p2[idx]), axis=1))]), axis=0)
                dist_2 = np.max(np.array([np.sqrt(np.sum(np.square(all_int_2 - p1[idx]), axis=1)),
                                          np.sqrt(np.sum(np.square(all_int_2 - p2[idx]), axis=1))]), axis=0)


                find_1 = np.where((dist_1 <= line_length[idx]) & (z_1 < np.squeeze(cy_shape[j, 1])))[0]
                find_2 = np.where((dist_2 <= line_length[idx]) & (z_2 < np.squeeze(cy_shape[j, 1])))[0]

                collide_line = np.unique(np.concatenate((find_temp, find_1, find_2)))
                count += len(collide_line)

                if ground:
                    another_collide_line = np.unique(np.concatenate((find_1, find_2)))
                    if len(another_collide_line) > 0:
                        all_int = np.abs(all_int_1[another_collide_line, 0:2] - all_int_2[another_collide_line, 0:2])
                        ground_penalties += ground_level[j] * np.sum(np.sqrt(all_int[:, 0]**2 + all_int[:, 1]**2))
        if ground:
            return ground_penalties
        else:
            return count

    def check_airspace(self, path, spatial_index, airspace_geoms):
        count = 0
        for i in range(len(path) - 1):
            path_segment = LineString([path[i], path[i + 1]])
            candidate_airspaces = list(spatial_index.intersection(path_segment.bounds))
            for idx in candidate_airspaces:
                if path_segment.intersects(airspace_geoms[idx]):
                    count += 1
                    break
        return count
    
    def check_ground(self, path, city_spatial_index, city_geoms, ground_level):
        ground_penalties = 0
        for i in range(len(path) - 1):
            path_segment = LineString([path[i], path[i + 1]])
            candidate_airspaces = list(city_spatial_index.intersection(path_segment.bounds))
            for j, idx in enumerate(candidate_airspaces):
                intersection = path_segment.intersection(city_geoms[idx])
                if not intersection.is_empty:
                    ground_penalties += intersection.length * ground_level[j]
                    break
        return ground_penalties

    def line_grid_val(self, p1, p2, bound_x, bound_y):
        x = bound_x + 1
        y = bound_y + 1
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - m * p1[0]
        mb = np.array([m, b])

        def lxmb(x, mb):
            return mb[0] * x + mb[1]

        def hix(y, mb):
            return np.array([(y - mb[1]) / mb[0], y])

        def vix(x, mb):
            return np.array([x, lxmb(x, mb)])

        hrz = hix(y, mb).T
        vrt = vix(x, mb).T
        hvix = np.vstack((hrz, vrt))

        if m > 0:
            if p1[1] < p2[1]:
                exbd = np.where((hvix[:, 1] < p1[1]) | (hvix[:, 1] > p2[1]))[0]
            else:
                exbd = np.where((hvix[:, 1] > p1[1]) | (hvix[:, 1] < p2[1]))[0]
        else:
            if p1[1] > p2[1]:
                exbd = np.where((hvix[:, 1] < p2[1]) | (hvix[:, 1] > p1[1]))[0]
            else:
                exbd = np.where((hvix[:, 1] > p2[1]) | (hvix[:, 1] < p1[1]))[0]

        hvix = np.delete(hvix, exbd, axis=0)
        hvix = np.unique(hvix, axis=0)

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        idx = np.argsort(hvix[:, 0])
        hvix = hvix[idx, :]

        line_segment = hvix

        diff = line_segment[0:-1, :] - line_segment[1:, :]
        line_length = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)

        if m > 0:
            grid_x = (np.ceil((hvix[:, 0] - x[0]) / dx)+bound_x[0]/3).astype(int)
            last_x = (np.ceil((p2[0] - x[0]) / dx)+bound_x[0]/3).astype(int)
            grid_y = (np.ceil((hvix[:, 1] - y[0]) / dy)+bound_y[0]/3).astype(int)
            last_y = (np.ceil((p2[1] - y[0]) / dy)+bound_y[0]/3).astype(int)
        else:
            grid_x = (np.ceil((hvix[:, 0] - x[0]) / dx)+bound_x[0]/3).astype(int)
            last_x = (np.ceil((p2[0] - x[0]) / dx + 1)+bound_x[0]/3).astype(int)
            grid_y = (np.floor((hvix[:, 1] - y[0]) / dy + 1)+bound_y[0]/3).astype(int)
            last_y = (np.ceil((p2[1] - y[0]) / dy)+bound_y[0]/3).astype(int)

        grid_pos = np.vstack((np.hstack((grid_y, last_y)), np.hstack((grid_x, last_x))))

        return line_length, grid_pos-1, hvix

    def get_fitness(self, population, spatial_index, airspace_geoms, W, wind_direction, sfip, cape, brn, city_spatial_index, city_geoms,
                    ground_level, range_min, range_max, left_bottom_corner, bound_x_enu, bound_y_enu, long_range):
        turn_angle = []
        constraint_violation = np.zeros((len(population), 4))
        penalties = []
        weather_penalties = []
        ground_penalties = []

        max_turn_angle = 90
        bound_x = np.array([range_min[0], range_max[0]])
        bound_y = np.array([range_min[1], range_max[1]])


        population_enu = np.zeros_like(population)
        for i in range(np.shape(population)[1]):
            x, y, z = pm.geodetic2enu(
                population[:, i, 0],
                population[:, i, 1],
                np.zeros(population.shape[0]),
                left_bottom_corner[0],
                left_bottom_corner[1],
                0
            )
            population_enu[:, i, :] = np.column_stack((x / 1000, y / 1000, np.zeros_like(x)))

        for i in range(0, np.shape(population)[0]):
            temp_enu = population_enu[i, 1:] - population_enu[i, 0:-1]
            line_length = np.sqrt(np.sum(np.square(temp_enu), axis=1))

            temp = population[i, 1:] - population[i, 0:-1]
            turn_angle.append(
                np.diagonal(np.matmul(temp[0:-1, 0:2], (temp[1:, 0:2]).T)) / (np.sqrt(np.square(temp[0:-1, 0]) +
                                                                                      np.square(temp[0:-1, 1])) * np.sqrt(
                    np.square(temp[1:, 0]) + np.square(temp[1:, 1]))))
            constraint_violation[i, 0] += np.sum(np.cos(np.radians(max_turn_angle)) > turn_angle[i])

            constraint_violation[i, 1] = 0

            count = self.check_airspace(population[i], spatial_index, airspace_geoms)
            # ground_penalties.append(
            #     self.check_cylinders(p1, p2, ground_risk_locations, ground_risk_shape, ground_level, True))
            if not long_range:
              ground_penalties.append(self.check_ground(population[i], city_spatial_index, city_geoms, ground_level))
            else:
              ground_penalties.append(0.0)

            constraint_violation[i, 2] += count

            constraint_violation[i, 3] += np.sum(population[i, :, 0] > bound_x[1])
            constraint_violation[i, 3] += np.sum(population[i, :, 1] > bound_y[1])
            constraint_violation[i, 3] += np.sum(population[i, :, 0] < bound_x[0])
            constraint_violation[i, 3] += np.sum(population[i, :, 1] < bound_y[0])

            time_segment = []
            weather_segment = 0
            all_line_segment_length = 0

            for j in range(0, np.shape(population)[1] - 1):
                loc_1 = np.squeeze(population[i, j, 0:2])
                loc_2 = np.squeeze(population[i, j + 1, 0:2])

                [line_segment_length, grid_pos, hvix] = self.line_grid_val(np.squeeze(population_enu[i, j, 0:2]), np.squeeze(population_enu[i, j + 1, 0:2]),
                                                                           bound_x_enu, bound_y_enu)

                all_line_segment_length += np.sum(line_segment_length)
                hvix_not_empty = hvix.size > 0 
                loc_x_min = min(loc_1[0], loc_2[0])
                loc_x_max = max(loc_1[0], loc_2[0])
                loc_y_min = min(loc_1[1], loc_2[1])
                loc_y_max = max(loc_1[1], loc_2[1])

                condition = (hvix_not_empty and
                             loc_x_min >= bound_x[0] and loc_x_max <= bound_x[-1] and
                             loc_y_min >= bound_y[0] and loc_y_max <= bound_y[-1])

                # Code to execute if the condition is True
                if np.linalg.norm(loc_2 - loc_1) < 1e-6:
                    time_segment.append(1)
                elif condition:
                    sfip_part = sfip[grid_pos[0, 1:-1], grid_pos[1, 1:-1]] * line_segment_length
                    cape_part = cape[grid_pos[0, 1:-1], grid_pos[1, 1:-1]] * line_segment_length
                    brn_part = brn[grid_pos[0, 1:-1], grid_pos[1, 1:-1]] * line_segment_length

                    w_part = W[grid_pos[0], grid_pos[1]]
                    wind_direction_1 = wind_direction[grid_pos[0], grid_pos[1]]
                    # wind_direction_2 = np.stack((np.sin(np.radians(wind_direction_1)), np.cos(np.radians(wind_direction_1)))).T
                    wind = np.stack(
                        (w_part * np.sin(np.radians(wind_direction_1)), w_part * np.cos(np.radians(wind_direction_1)))).T
                    ground_speed = population[i, j + 1, 0:2] - population[i, j, 0:2]
                    ground_speed_normalized = ground_speed / np.linalg.norm(ground_speed)
                    D = ground_speed_normalized[0] * wind[:, 0] + ground_speed_normalized[1] * wind[:, 1] + np.sqrt(
                        np.square(ground_speed_normalized[0] * wind[:, 0] + ground_speed_normalized[1] * wind[:,
                                                                                                         1]) + 6400 - wind[
                                                                                                                      :,
                                                                                                                      0] ** 2 - wind[
                                                                                                                                :,
                                                                                                                                1] ** 2)
                    airspeed = (np.expand_dims(D, axis=1) * np.expand_dims(ground_speed_normalized, axis=0) - wind)
                    ground_speed = wind + airspeed
                    ground_speed = ground_speed[1:-1]
                    time_segment.append(line_segment_length / (1.85200 * np.sqrt(np.sum(np.square(ground_speed), axis=1))))
                    weather_segment += np.sum(sfip_part + cape_part + brn_part)
                else:
                    time_segment.append(1)

            if len(time_segment) > 0:
                penalties.append([np.sum(line_length, axis=0), np.sum(np.hstack(time_segment))])
                # weather_penalties.append(weather_segment)
                weather_penalties.append(0)
            else:
                penalties.append([np.sum(line_length, axis=0), 0])
                weather_penalties.append(0)

        penalties = np.stack(penalties)
        weather_penalties = np.stack(weather_penalties)
        ground_penalties = np.stack(ground_penalties)

        return constraint_violation, penalties, weather_penalties, ground_penalties

    def get_angle(self, u, v):
        x = np.arctan2(v, u) * 180 / np.pi
        if x < 0:
            x = abs(x) + 90
        else:
            x = 90 - x
            if x < 0:
                x = 360 + x
        return x

    def get_wind(self, data_1, data_2, new=False):
        if not new:
            u = np.squeeze(data_1[:, :, 0])
            v = np.squeeze(data_1[:, :, 1])
        else:
            u = data_1
            v = data_2
        M = np.sqrt(np.square(u) + np.square(v))
        wind_direction = np.ones(np.shape(u))
        for i in range(0, u.shape[0]):
            for j in range(0, u.shape[1]):
                wind_direction[i, j] = self.get_angle(u[i, j], v[i, j])

        return M, wind_direction

    def create_cylinder(self, radius, height=1, num_points=100):
        theta = np.linspace(0, 2 * np.pi, num_points)
        z = np.array([0, height])

        X = np.outer(np.cos(theta), np.ones(2)) * radius  # (num_points, 2)
        Y = np.outer(np.sin(theta), np.ones(2)) * radius  # (num_points, 2)
        Z = np.outer(np.ones(num_points), z)  # (num_points, 2)

        return X, Y, Z
    
    

    def update_population(self, population, velocity, p, g): # population:(N, n_points, 3)
        w = 0.8
        c1 = 1.496
        c2 = 1.496

        rp = self.rng.random((np.shape(population)))
        rg = self.rng.random((np.shape(population)))
        g = np.tile(g, (np.shape(population)[0], 1, 1))

        velocity = w*velocity + c1*rp*(p - population) + c2*rg*(g - population)
        return population+velocity

    def plot_path(self, best_path, airspace_geo, all_cities_geo, sfip, brn, cape, lons, lats, land_mark, n_points, long_range, xylims, map_source):
        fig, ax = plt.subplots(figsize=(10, 10))
        center_x, center_y, half_size, ratio = xylims
        ax.set_xlim(center_x - half_size, center_x + half_size)
        ax.set_ylim(center_y - half_size, center_y + half_size)

        # ax.set_aspect('equal', adjustable='box')

        airspace_gdf_list = []
        for area in airspace_geo:
            geometry = LineString([(point[1], point[0]) for point in area])
            airspace_gdf_list.append(gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326"))
        
        if airspace_gdf_list:
            airspace_gdf = gpd.GeoDataFrame(pd.concat(airspace_gdf_list, ignore_index=True), crs="EPSG:4326")
            airspace_gdf.plot(ax=ax, color='#003366', linewidth=2)
        
        cities_gdf_list = []
        for city in all_cities_geo:
            geometry = LineString([(point[1], point[0]) for point in city])
            cities_gdf_list.append(gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326"))
        
        if not long_range:
          if cities_gdf_list:
              cities_gdf = gpd.GeoDataFrame(pd.concat(cities_gdf_list, ignore_index=True), crs="EPSG:4326")
              cities_gdf.plot(ax=ax, color='#8B0000')
        
        points = [(point[1], point[0]) for point in best_path]
        points_gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in points], crs="EPSG:4326")
        points_gdf.iloc[[0]].plot(ax=ax, color='red', marker='*', markersize=400, edgecolor='black', linewidth=2, label='Take-Off Point', zorder=3)
        points_gdf.iloc[[-1]].plot(ax=ax, color='purple', marker='D', markersize=100, edgecolor='black', linewidth=2, label='Landing Point', zorder=3)
        
        path_gdf = gpd.GeoDataFrame(geometry=[LineString(points)], crs="EPSG:4326")
        path_gdf.plot(ax=ax, color='black', linewidth=3)
        for j in range(len(land_mark)):
            idx = int((n_points+1)*(j+1))
            if idx < len(points_gdf):
                way_point = points_gdf.iloc[[idx]]
                label = 'Mission Point' if j == 0 else None
                way_point.plot(ax=ax, color='black', markersize=150, label=label, zorder=2)

        airspace = Circle((0,0), radius=2, edgecolor='#003366', facecolor='none', linewidth=2, fill=False)
        populated_area = Rectangle((0,0), width=1, height=1, edgecolor='#8B0000', facecolor='none', linewidth=2, fill=False)
        def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            return Circle((xdescent + width/2, ydescent + height/2), min(width, height)/2)
        legend_handler = {airspace: HandlerPatch(patch_func=make_legend_circle)}
        weather_risk = Patch(facecolor='red', alpha=0.3, edgecolor='none', label='Weather Risk')

        ctx.add_basemap(ax, crs=path_gdf.crs, source=map_source)
        
        lon_min, lon_max = center_x - half_size, center_x + half_size
        lat_min, lat_max = center_y - half_size*ratio, center_y + half_size*ratio
        lons = np.where(lons > 180, lons - 360, lons)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        combined_mask = lon_mask & lat_mask
        valid_rows, valid_cols = np.where(combined_mask)
        lat_start, lat_end = valid_rows.min(), valid_rows.max() + 1
        lon_start, lon_end = valid_cols.min(), valid_cols.max() + 1
        
        sfip_crop = sfip[lat_start:lat_end, lon_start:lon_end]
        brn_crop = brn[lat_start:lat_end, lon_start:lon_end]
        cape_crop = cape[lat_start:lat_end, lon_start:lon_end]
        weather_data = sfip_crop + brn_crop + cape_crop
        data_min, data_max = weather_data.min(), weather_data.max()
        weather_normalized = ((weather_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        
        upsample_factor = 10
        pil_image = Image.fromarray(weather_normalized)
        max_dim = max(weather_normalized.shape[0], weather_normalized.shape[1])
        upsampled_image = pil_image.resize((max_dim * upsample_factor, max_dim * upsample_factor), Image.LANCZOS)
        
        weather_heat = np.array(upsampled_image).astype(np.float32)
        weather_heat = weather_heat / 255.0 * (data_max - data_min) + data_min
        ax.imshow(weather_heat, cmap='hot', alpha=0.3, extent=[center_x - half_size, center_x + half_size, center_y - half_size, center_y + half_size])

        ax.legend([airspace, populated_area, weather_risk], ["Controlled Airspace", "Populated Area", "Weather Risk"], handler_map=legend_handler, loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=18, fancybox=True, shadow=True)
        plt.axis('off')
        plt.savefig('./temp/fig_path.png', bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
