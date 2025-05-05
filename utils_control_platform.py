import requests
import time
import json
from pprint import pprint
from math import sin, asin, cos, tan, atan2, degrees, radians

def get_point_at_distance(lat1, lon1, d, bearing, R=6371000):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial in m
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}m from initial, in degrees
    """
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    a = radians(bearing)
    lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
    lon2 = lon1 + atan2(
        sin(a) * sin(d/R) * cos(lat1),
        cos(d/R) - sin(lat1) * sin(lat2)
    )
    return (degrees(lat2), degrees(lon2))

def createLocalCircuit(runway):
    """
    Creates a local circuit pattern including multiple loops of the chosen runway with the "Kidney Bean" profile
    """
    # Get information about runway (Lat/Lng in decimal degrees)
    start_lat = runway['lat']
    start_lng = runway['lng']
    start_alt = runway['alt']
    runway_heading_deg = runway['heading']
    traffic_pattern = runway['traffic_pattern']
    
    if traffic_pattern == 'Left':
        pattern_right_angle = -90
        pattern_inner_angle = -12
    elif traffic_pattern == 'Right':
        pattern_right_angle = 90
        pattern_inner_angle = 12
    
    # Standard circuit pattern parameters for ULTRA
    landing_glide_slope = 5 # degrees
    base_glide_slope = 3 # degrees
    
    # DISTANCES - ALL IN METRES
    takeoff_dis_m = 750
    takeoff_to_crosswind_dis_m = 650
    crosswind_dis_m = 1000
    downwind_dis_m = (takeoff_dis_m + takeoff_to_crosswind_dis_m) * 2 # this assumes aircraft starts on center on runway - which it does not!
    base_dis_m = crosswind_dis_m
    upwind_mid_distance_m =  (downwind_dis_m / 2) / cos(radians(pattern_inner_angle)) # the kidney bean!
    final_dis_m = 1750
    
    # ALTITUDES - ALL IN METRES - AGL
    circuit_alt_m = start_alt + 350 # (1000ft AGL)
    takeoff_alt_m = start_alt + 125
    crosswind_start_alt_m = takeoff_alt_m + 50
    downwind_start_alt_m = crosswind_start_alt_m + 50
    final_approach_alt_m = start_alt + (final_dis_m * tan(radians(landing_glide_slope)))
    base_approach_alt_m = final_approach_alt_m + (base_dis_m * tan(radians(base_glide_slope)))

    # Waypoint 0
    # Takeoff Waypoint
    waypoint_0_lat, waypoint_0_lng = get_point_at_distance(start_lat, start_lng, takeoff_dis_m, runway_heading_deg)
    waypoint_0 = {
                    "index": 0,
                    "location":
                        {
                            "latDeg": waypoint_0_lat,
                            "lngDeg": waypoint_0_lng,
                            "altM": takeoff_alt_m
                        },
                    "type": "TakeOff"
                }
    
    # Waypoint 1
    # Crosswind start Waypoint
    waypoint_1_lat, waypoint_1_lng = get_point_at_distance(waypoint_0_lat, waypoint_0_lng, takeoff_to_crosswind_dis_m, runway_heading_deg)
    waypoint_1 = {
                    "index": 1,
                    "location":
                        {
                            "latDeg": waypoint_1_lat,
                            "lngDeg": waypoint_1_lng,
                            "altM": crosswind_start_alt_m
                        },
                    "type": "Waypoint",
                    "commands": [
                        {
                            "commandType": "LandStart",
                            "index": 1
                        }
                    ]
                }
    
    # Waypoint 2
    # Downwind start Waypoint
    crosswind_heading_deg = runway_heading_deg + pattern_right_angle
    waypoint_2_lat, waypoint_2_lng = get_point_at_distance(waypoint_1_lat, waypoint_1_lng, crosswind_dis_m, crosswind_heading_deg)
    waypoint_2 = {
                    "index": 2,
                    "location":
                        {
                            "latDeg": waypoint_2_lat,
                            "lngDeg": waypoint_2_lng,
                            "altM": downwind_start_alt_m
                        },
                    "type": "Waypoint"
                }
    
    # Waypoint 3
    # Base start Waypoint
    downwind_heading_deg = crosswind_heading_deg + pattern_right_angle
    waypoint_3_lat, waypoint_3_lng = get_point_at_distance(waypoint_2_lat, waypoint_2_lng, downwind_dis_m, downwind_heading_deg)
    waypoint_3 = {
                    "index": 3,
                    "location":
                        {
                            "latDeg": waypoint_3_lat,
                            "lngDeg": waypoint_3_lng,
                            "altM": circuit_alt_m
                        },
                    "type": "Waypoint"
                }
    
    # Waypoint 4
    # Upwind start Waypoint
    base_heading_deg = downwind_heading_deg + pattern_right_angle
    waypoint_4_lat, waypoint_4_lng = get_point_at_distance(waypoint_3_lat, waypoint_3_lng, base_dis_m, base_heading_deg)
    waypoint_4 = {
                    "index": 4,
                    "location":
                        {
                            "latDeg": waypoint_4_lat,
                            "lngDeg": waypoint_4_lng,
                            "altM": circuit_alt_m
                        },
                    "type": "Waypoint"
                }
    
    # Waypoint 5
    # Mid-Upwind Waypoint
    mid_upwind_heading_deg = base_heading_deg + (pattern_right_angle + pattern_inner_angle)
    waypoint_5_lat, waypoint_5_lng = get_point_at_distance(waypoint_4_lat, waypoint_4_lng, upwind_mid_distance_m - 50, mid_upwind_heading_deg)
    waypoint_5 = {
                    "index": 5,
                    "location":
                        {
                            "latDeg": waypoint_5_lat,
                            "lngDeg": waypoint_5_lng,
                            "altM": circuit_alt_m
                        },
                    "type": "Waypoint"
                }
    
    # Waypoint 6
    # Crosswind at Circuit Altitude Start Waypoint
    crosswind_2_heading_deg = mid_upwind_heading_deg - (2 * pattern_inner_angle)
    waypoint_6_lat, waypoint_6_lng = get_point_at_distance(waypoint_5_lat, waypoint_5_lng, upwind_mid_distance_m - 50, crosswind_2_heading_deg)
    waypoint_6 = {
                    "index": 6,
                    "location":
                        {
                            "latDeg": waypoint_6_lat,
                            "lngDeg": waypoint_6_lng,
                            "altM": circuit_alt_m
                        },
                    "type": "Waypoint"
                }
    
    # Waypoint 7
    # Downwind at Circuit Altitude Start Waypoint
    waypoint_7_lat, waypoint_7_lng = get_point_at_distance(waypoint_6_lat, waypoint_6_lng, crosswind_dis_m, crosswind_heading_deg)
    waypoint_7 = {
                    "index": 7,
                    "location":
                        {
                            "latDeg": waypoint_7_lat,
                            "lngDeg": waypoint_7_lng,
                            "altM": circuit_alt_m
                        },
                    "type": "Waypoint",
                    "commands": [
                        {
                        "commandType": "Jump",
                        "waypointIndex": 3,
                        "numRepeats": 0,
                        "index": 9
                        }
                     ] 
                }
    
    # Waypoint 8
    # Base at Approach Altitude Start Waypoint
    waypoint_8_lat, waypoint_8_lng = get_point_at_distance(waypoint_7_lat, waypoint_7_lng, downwind_dis_m, downwind_heading_deg)
    waypoint_8 = {
                    "index": 8,
                    "location":
                        {
                            "latDeg": waypoint_8_lat,
                            "lngDeg": waypoint_8_lng,
                            "altM": base_approach_alt_m
                        },
                    "type": "Waypoint"
                }
    
    # Waypoint 9
    # Final at Approach Altitude Start Waypoint
    waypoint_9_lat, waypoint_9_lng = get_point_at_distance(waypoint_8_lat, waypoint_8_lng, base_dis_m, base_heading_deg)
    waypoint_9 = {
                    "index": 9,
                    "location":
                        {
                            "latDeg": waypoint_9_lat,
                            "lngDeg": waypoint_9_lng,
                            "altM": final_approach_alt_m
                        },
                    "type": "Waypoint"
                }
    
    # Waypoint 10
    # Final at Approach Altitude Start Waypoint
    waypoint_10_lat, waypoint_10_lng = get_point_at_distance(waypoint_9_lat, waypoint_9_lng, final_dis_m, runway_heading_deg)
    waypoint_10 = {
                    "index": 10,
                    "location":
                        {
                            "latDeg": waypoint_10_lat,
                            "lngDeg": waypoint_10_lng,
                            "altM": start_alt
                        },
                    "type": "Land",
                    "abortHeightM": 50
                }
    
    waypoints = [waypoint_0, waypoint_1, waypoint_2, waypoint_3, waypoint_4, waypoint_5,
                waypoint_6, waypoint_7, waypoint_8, waypoint_9, waypoint_10]
    
    return waypoints

####### REST API #######
# Ground Control, Platform Control, and Cloud Control all allow data to be accessed via API
# This is intended to allow developers to build products on top of Distributed Controls ecosystem
# This example shows how to access some of them from python, full documentation can be found in the user
# manual of each system.

# Use the correct url for the relevant DA product:
# Ground Control -> "http://127.0.0.1:5001" (API & SignalR)
# Platform Control -> "http://127.0.0.1:5000" (API & SignalR)
# Cloud Control -> "https://cloud.distributed-avionics.com" (API & SignalR)
# Traffic Server -> "https://traffic-h-wndr.distributed-avionics.com" (SignalR)
# Terrain Server -> "https://terrain.distributed-avionics.com" (API)
# Airspace Server -> "https://airspace-g-wndr.distributed-avionics.com" (API)
# Weather Server -> "https://weather-h-wndr.distributed-avionics.com" (API)

####### GETTING AUTHENTICATION #######
# Safety Critical API's are locked to users with Admin, or Operator privillidges, you need a token to access them
# and you also need to know the platformId of your system

# For this example we are connecting to a locally running Ground Control
# Username and Password and generic, and purely used to get an Authentication Token
# CONNECTION_URL = "http://127.0.0.1:5001"
# USERNAME = "Admin"
# PASSWORD = "Distributed"
CONNECTION_URL = "https://cloud.distributed-avionics.com"
USERNAME = "ChuhaoDeng"
PASSWORD = "deng113_Platform"


mission_file = "mission.json"


# runway object should have lat, lng, alt, for runway center point 
KZRL_090_runway = {
            "lat": 40.947514, 
            "lng": -87.189650,
            "alt": 213,
            "heading": 90,
            "traffic_pattern": 'Left'
        }

KZRL_180_runway = {
            "lat": 40.952748, 
            "lng": -87.181517,
            "alt": 213,
            "heading": 180,
            "traffic_pattern": 'Left'
        }

KZRL_270_runway = {
            "lat": 40.947566, 
            "lng": -87.183812,
            "alt": 213,
            "heading": 270,
            "traffic_pattern": 'Right'
        }



KZRL_360_runway = {
            "lat": 40.943075, 
            "lng": -87.181479,
            "alt": 213,
            "heading": 360,
            "traffic_pattern": 'Right'
        }

KLAF_050_runway = {
            "lat": 40.408403, 
            "lng": -86.938708,
            "alt": 183,
            "heading": 50,
            "traffic_pattern": 'Left'
        }

KLAF_100_runway = {
            "lat": 40.412925,
            "lng": -86.942407,
            "alt": 183,
            "heading": 99,
            "traffic_pattern": 'Left'
        }

KLAF_230_runway = {
            "lat": 40.415044, 
            "lng": -86.928533,
            "alt": 183,
            "heading": 230,
            "traffic_pattern": 'Right'
        }

KLAF_280_runway = {
            "lat": 40.411214, 
            "lng": -86.928389,
            "alt": 183,
            "heading": 280,
            "traffic_pattern": 'Right'
        }

KRID_60_runway = {
            "lat": 39.752487, 
            "lng": -84.848504,
            "alt": 346,
            "heading": 60,
            "traffic_pattern": 'Left'
        } # Richmond

KBAK_50_runway = {
            "lat": 39.255837, 
            "lng": -85.904182,
            "alt": 200,
            "heading": 50,
            "traffic_pattern": 'Left'
        } # Columnbus


KEVV_360_runway = {
            "lat": 38.031774, 
            "lng": -87.534388,
            "alt": 116.7,
            "heading": 360,
            "traffic_pattern": 'Left'
        } # Evansville

KIND_50_runway = {
            "lat": 39.71735330835, 
            "lng": -86.30684189165,
            "alt": 116.7,
            "heading": 50,
            "traffic_pattern": 'Left'
        } # Indy

def create_json(input_file, mission_file, starting_idx):    
    with open(input_file, 'r') as file:
        lines = file.readlines()

    all_latitude = []
    all_longitude = []

    # Parse the data
    waypoints = []
    for i, line in enumerate(lines):
        lat, lon, alt = map(float, line.split())
        all_latitude.append(lat)
        all_longitude.append(lon)
        
        if i == 0:
            waypoint_type = "TakeOff"
        elif i == 3:
            waypoint_type = "LoiterTurns"
        elif i == len(lines) - 1:
            waypoint_type = "Land"
        else:
            waypoint_type = "Waypoint"
        
        waypoint = {
            "index": i + starting_idx,
            "location": {
                "latDeg": lat,
                "lngDeg": lon,
                "altM": 336.105161170367
            },
            "type": waypoint_type
        }
        
        if waypoint_type == "LoiterTurns":
            waypoint["numTurns"] = 5
            waypoint["radiusM"] = 30
        
        if waypoint_type == "Land":
            waypoint["abortHeightM"] = 91
        
        waypoints.append(waypoint)

    # Create the mission dictionary
    mission = {
        "waypoints": waypoints,
        "forceWrite": True
    }

    # Write the JSON to a file
    with open(mission_file, 'w') as file:
        json.dump(mission, file, indent=2)

    print(f"{mission_file} file has been created successfully.")
    time.sleep(3)
    if abs(min(all_latitude) - max(all_latitude)) > 8 or abs(min(all_longitude) - max(all_longitude)) > 8:
        long_range = True
    else:
        long_range = False

    return i + starting_idx, long_range


def start_sim(sim_number, headers):
    """
    Receives an encounter scenario and starts a simulator that is relevant for this.
    Returns the expected sim ID
    """
   
    print("Starting Simulator " + str(sim_number))
    payload = {
        "simulationType": "ProductionSim-Peregrine",
        "numberOfInstances": 1,
        "prefix": "DAATest" + str(sim_number),
        "location": {                                 # Spawn location for the aircraft
                    "latDeg": 40.411,
                    "lngDeg": -86.9272,
                    "altM": 213.36
                },                                 
        "headingDeg": 280                                 # Spawn heading for the aircraft
    }

    response = requests.post(CONNECTION_URL + '/app/Simulation/addBatchSimulation', headers=headers, json=payload)
   
    # Get the Sim ID
    simulator_dict = json.loads(response.text)
    sim_ID = simulator_dict['instanceIds'][0]
   
    # Report to user the status of the simulator
    print("Started Simulator with ID:   " + str(sim_ID))
   
    return sim_ID


# ARMING THE AIRCRAFT
def set_arm_for_sim(platform_ID, headers):
    """
    Receives a platform ID and will arm the aircraft
    """
    response_arming = requests.post(CONNECTION_URL + '/autopilot/Command/arm?platformID=' + platform_ID, headers=headers)
    while response_arming.status_code != 200:
        response_arming = requests.post(CONNECTION_URL + '/autopilot/Command/arm?platformID=' + platform_ID, headers=headers)
        time.sleep(1)
    print("Simulator Armed: " + platform_ID)
   

# CHANGING MODE TO AUTO
def set_auto_mode_for_sim(platform_ID, headers):
    """
    Receives a platform ID and will set the mode to Auto
    """
    payload = {"platformId": platform_ID, "mode": "Auto"}
    response_mode = requests.post(CONNECTION_URL + '/autopilot/Mode/setMode', json = payload, headers=headers)
    while response_mode.status_code != 200:
        response_mode = requests.post(CONNECTION_URL + '/autopilot/Mode/setMode', json = payload, headers=headers)
        time.sleep(1)
    print("Mission Started!")


def execute_pipeline(input_file):


# Get authentication token
    print("Getting Token...")
    payload = {"username": USERNAME, "password": PASSWORD }
    response = requests.post(CONNECTION_URL + '/authentication/authenticate', json = payload)
    token = response.json()["token"]
    print("Got Token: " + token, "\n")

    # Create authorised header
    headers = {"Authorization": "Bearer "+ token}

    # response = requests.get(CONNECTION_URL + '/app/account/getPlatforms',headers=headers)
    # firstPlatformInfo = response.json()[0]
    # active_platform_ID = firstPlatformInfo["id"]
    # print("Found a platform with Id: %s" % active_platform_ID, "\n")


    ####### STARTING SIMULATOR ##################

    # Start a simulator with the correct starting conditions
    sim_ID = start_sim(0, headers)
    time.sleep(1)
    
    print("Please wait for the Sims to start...")
    
    # Check to see if required simulators have started
    ready = False
    while not ready:
        response = requests.get(CONNECTION_URL + '/app/Simulation/getSimulations', headers=headers)
        response_dict = json.loads(response.text)
    
        active_platform_ID = "00000000-0000-0000-0000-000000000000"
        active_sim_ID = ""
    
        active_platform_ID = response_dict[0]['platformId']
        active_sim_ID = response_dict[0]['id']
        
        # Sim will be given platform ID once fully initialised, this check ensure Sims are up and running before continuing
        if active_platform_ID != "00000000-0000-0000-0000-000000000000":
            ready = True
            print("Platform IDs Started: " + str(active_platform_ID))
        else:
            ready = False
        time.sleep(1)
    
    print("Sims Ready!")
    time.sleep(5) # some settling time for the simulators?

    ####### SETTING AND GETTING A MISSION #######

    # CHANGE TAKEOFF AIRPORT HERE
    take_off_circuit = createLocalCircuit(KLAF_050_runway)
    starting_idx = len(take_off_circuit) - 2
    
        # Create json file from txt
    ending_idx, long_range = create_json(input_file=input_file, mission_file=mission_file, starting_idx=starting_idx) - 1

    if not long_range:
        # CHANGE LANDING AIRPORT HERE
        take_off_circuit = createLocalCircuit(KIND_50_runway)
        land_circuit = createLocalCircuit(KLAF_050_runway) # Richmond: KRID

        for i, waypoint in enumerate(land_circuit):
            waypoint["index"] = i + ending_idx

        # Set mission using token
        print("Setting Mission...")
        with open(f'{mission_file}', 'r') as f:
            data = json.load(f)

        payload = {
                    "platformId": active_platform_ID,
                    "waypoints": take_off_circuit[:-1] + data["waypoints"][1: -1] + land_circuit[1:],
                    "forceWrite": True
                }    
        
        payload["waypoints"][-4]["commands"][0]["waypointIndex"] = payload["waypoints"][-4]["index"] - 4
    else:
        # Need to change it to multiple airports
        # CHANGE LANDING AIRPORT HERE
        take_off_circuit = createLocalCircuit(KIND_50_runway)
        land_circuit = createLocalCircuit(KLAF_050_runway) # Richmond: KRID

        for i, waypoint in enumerate(land_circuit):
            waypoint["index"] = i + ending_idx

        # Set mission using token
        print("Setting Mission...")
        with open(f'{mission_file}', 'r') as f:
            data = json.load(f)

        payload = {
                    "platformId": active_platform_ID,
                    "waypoints": take_off_circuit[:-1] + data["waypoints"][1: -1] + land_circuit[1:],
                    "forceWrite": True
                }    
        
        payload["waypoints"][-4]["commands"][0]["waypointIndex"] = payload["waypoints"][-4]["index"] - 4

    with open(f'test.json', 'w') as outfile:
        json.dump(payload, outfile, indent=2)
    # print(data)
        

    count = 1
    response = requests.post(CONNECTION_URL + '/autopilot/Mission/setMission', json = payload, headers=headers)
    while response.status_code != 200:
        response = requests.post(CONNECTION_URL + '/autopilot/Mission/setMission', json = payload, headers=headers)
        print(f"Counter: {count}")
        count += 1
        time.sleep(0.5)

    print("Mission Updated for platform ID: " + active_platform_ID)
    print(response.text) # Prints the code returned by the API
    print(response.ok, "\n")
    
    # time.sleep(3)

    # set_arm_for_sim(payload['platformId'], headers)
    # set_auto_mode_for_sim(payload['platformId'], headers)

    # pressedEnter = None
    # while pressedEnter != "":
    #     pressedEnter = input("")
    #     time.sleep(1)

    # payload = [active_sim_ID]
    # response = requests.post(CONNECTION_URL + '/app/Simulation/deleteSimulations', json = payload, headers=headers)
    # print("Closing Simulators!")