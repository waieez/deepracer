import math

# {
#     "all_wheels_on_track": Boolean,        # flag to indicate if the agent is on the track
#     "x": float,                            # agent's x-coordinate in meters
#     "y": float,                            # agent's y-coordinate in meters
#     "closest_objects": [int, int],         # zero-based indices of the two closest objects to the agent's current position of (x, y).
#     "closest_waypoints": [int, int],       # indices of the two nearest waypoints.
#     "distance_from_center": float,         # distance in meters from the track center 
#     "is_crashed": Boolean,                 # Boolean flag to indicate whether the agent has crashed.
#     "is_left_of_center": Boolean,          # Flag to indicate if the agent is on the left side to the track center or not. 
#     "is_offtrack": Boolean,                # Boolean flag to indicate whether the agent has gone off track.
#     "is_reversed": Boolean,                # flag to indicate if the agent is driving clockwise (True) or counter clockwise (False).
#     "heading": float,                      # agent's yaw in degrees
#     "objects_distance": [float, ],         # list of the objects' distances in meters between 0 and track_length in relation to the starting line.
#     "objects_heading": [float, ],          # list of the objects' headings in degrees between -180 and 180.
#     "objects_left_of_center": [Boolean, ], # list of Boolean flags indicating whether elements' objects are left of the center (True) or not (False).
#     "objects_location": [(float, float),], # list of object locations [(x,y), ...].
#     "objects_speed": [float, ],            # list of the objects' speeds in meters per second.
#     "progress": float,                     # percentage of track completed
#     "speed": float,                        # agent's speed in meters per second (m/s)
#     "steering_angle": float,               # agent's steering angle in degrees
#     "steps": int,                          # number steps completed
#     "track_length": float,                 # track length in meters.
#     "track_width": float,                  # width of the track
#     "waypoints": [(float, float), ]        # list of (x,y) as milestones along the track center
# }


def reward_function(params):
    # to progress, reward for moving towards waypoints
    reward = 0

    # raw params
    all_wheels_on_track = params['all_wheels_on_track']
    closest_waypoints = params['closest_waypoints']
    distance_from_center = params['distance_from_center']
    heading = params['heading']
    is_crashed = params['is_crashed']
    is_offtrack = params['is_offtrack']
    position = (params['x'], params['y'])
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    track_length = params['track_length']
    track_width = params['track_width']
    waypoints = params['waypoints']

    # derived_params
    num_waypoints = len(waypoints)

    current_waypoint_index = min(closest_waypoints)
    next_waypoint_index = max(closest_waypoints)

    current_waypoint = waypoints[current_waypoint_index]
    next_waypoint = waypoints[next_waypoint_index]

    distance_to_current_waypoint = euclidian_distance(position, current_waypoint)
    distance_to_next_waypoint = euclidian_distance(position, next_waypoint)

    # reward calculations

    # reward for advancing towards closest waypoint
    reward += .5 * (1 + current_waypoint_index) / (1 + distance_to_current_waypoint)

    # reward for advancing towards next waypoint
    reward += 1.5 * (1 + next_waypoint_index) / (1 + distance_to_next_waypoint)

    # reward for going fast
    reward += min(4, speed) * 2

    # scale reward by how centered the car is
    half_track_width = track_width / 2
    reward *= (half_track_width - distance_from_center) / half_track_width

    # reward for turning towards center line
    reward *= track_direction_factor(heading, current_waypoint, next_waypoint)

    if is_crashed:
        reward = -1000
    
    if is_offtrack:
        reward = -1000

    reward -= steps / 100

    return float(reward)



def euclidian_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )


def track_direction_factor(heading, a, b):
    x1, y1 = a
    x2, y2 = b
    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(y2 - y1, x2 - x1)
    # Convert to degree
    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    # Penalize the reward if the difference is too large
    DIRECTION_THRESHOLD = 10.0
    if direction_diff > DIRECTION_THRESHOLD:
        return .5
    
    return 1
