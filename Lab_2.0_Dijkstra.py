import os
import numpy as np
import heapq
import csv
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import time
import math
from Robotino_communication import connect_to_robotino, send_velocity

# --- NEW GLOBAL VARIABLES FOR OBJECT AND ROBOT RADII (in cm) ---
OBJECT_RADIUS = 18  # adjust as needed
ROBOT_RADIUS = 25  # adjust as needed


def setup_csv_logger(filename="trajectory_log.csv"):
    log_file = open(filename, mode='w', newline='')
    writer = csv.writer(log_file)
    # Write header
    writer.writerow([
        "frame",
        "robot_x_cm", "robot_y_cm",
        "target_x_cm", "target_y_cm",
        "path_length",
        "waypoints"  # semicolon-separated list of "x,y"
    ])
    return log_file, writer


def log_frame_data(writer, frame_count, robot_pos, target_pos, path):
    if robot_pos is None or target_pos is None:
        return  # skip if data is missing

    rx, ry = robot_pos
    tx, ty = target_pos
    path_len = len(path) if path else 0

    # Convert waypoints to a single string: "x1,y1;x2,y2;..."
    waypoints_str = ';'.join(f"{wp[0]:.2f},{wp[1]:.2f}" for wp in path)

    writer.writerow([
        frame_count,
        f"{rx:.2f}", f"{ry:.2f}",
        f"{tx:.2f}", f"{ty:.2f}",
        path_len,
        waypoints_str
    ])


# --- DIJKSTRA'S ALGORITHM IMPLEMENTATION ---
class DijkstraPlanner:
    def __init__(self, width, height, grid_size=20):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cell_w = width / grid_size
        self.cell_h = height / grid_size

    def world_to_grid(self, position):
        x, y = position
        gx = int(x / self.cell_w)
        gy = int(y / self.cell_h)
        gx = max(0, min(gx, self.grid_size - 1))
        gy = max(0, min(gy, self.grid_size - 1))
        return (gx, gy)

    def grid_to_world(self, grid_pos):
        gx, gy = grid_pos
        wx = (gx * self.cell_w) + (self.cell_w / 2)
        wy = (gy * self.cell_h) + (self.cell_h / 2)
        return (wx, wy)

    def plan(self, start_pos, target_pos, obstacles, object_radius=0, robot_radius=0):
        # Create grid (0 = free, 1 = obstacle)
        grid = np.zeros((self.grid_size, self.grid_size))

        start_node = self.world_to_grid(start_pos)
        target_node = self.world_to_grid(target_pos)

        # Safety margin = object radius + robot radius
        margin = object_radius + robot_radius

        # Mark obstacles based on distance from each object center
        for obs in obstacles:
            if obs[0] == 0 and obs[1] == 0:
                continue  # Skip undetected
            # For every cell, check distance to this object
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if grid[i][j] == 1:
                        continue  # already blocked
                    cell_center = self.grid_to_world((i, j))
                    dx = cell_center[0] - obs[0]
                    dy = cell_center[1] - obs[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= margin:
                        grid[i][j] = 1

        # Ensure start and target are free
        grid[start_node[0]][start_node[1]] = 0
        grid[target_node[0]][target_node[1]] = 0

        # Dijkstra's Algorithm (no heuristic, just cost from start)
        open_set = []
        heapq.heappush(open_set, (0, start_node))  # (cost, node)
        came_from = {}
        cost_so_far = {start_node: 0}

        while open_set:
            current_cost, current = heapq.heappop(open_set)

            if current == target_node:
                return self.reconstruct_path(came_from, current)

            # Checking for corners and up down left and right
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Added diagonal movement
                neighbor = (current[0] + dx, current[1] + dy)

                # Checking to see if the neighbors are inside the grid
                if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue

                    # Calculate movement cost (diagonal costs more)
                    if dx != 0 and dy != 0:
                        move_cost = 1.4  # Diagonal movement cost (sqrt(2))
                    else:
                        move_cost = 1  # Cardinal movement cost

                    new_cost = cost_so_far[current] + move_cost
                    #  Updating the neighbors. If first time or if cheaper
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost  # Updating the cost
                        came_from[neighbor] = current  # updating where is the current coming from
                        heapq.heappush(open_set, (new_cost, neighbor))  # push the new neighbor into the heap to check later

        return []  # No path

    def reconstruct_path(self, came_from, current):
        path = [self.grid_to_world(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self.grid_to_world(current))
        path.reverse()
        return path


# --- END DIJKSTRA IMPLEMENTATION ---
csv_log_file, csv_writer = setup_csv_logger("trajectory_log.csv")
frame_counter = 0
vid = cv.VideoCapture(0, cv.CAP_MSMF)
# vid = cv.VideoCapture("Mazes/2.mp4")
resTrack = (1600, 1200)
vid.set(cv.CAP_PROP_FRAME_WIDTH, resTrack[0])
vid.set(cv.CAP_PROP_FRAME_HEIGHT, resTrack[1])
vid.set(cv.CAP_PROP_FPS, 30)

dict = cv.aruco.DICT_5X5_100
dictionary = cv.aruco.getPredefinedDictionary(dict)

# Data containers
Field_ArMark_Center = [(0, 0), (0, 0), (0, 0), (0, 0)]
ObjectMarkers = [(0, 0) for _ in range(6)]
ObjectPosition = [(0, 0) for _ in range(6)]
RobotMarker = [(0, 0)]
RobotPosition = [(0, 0)]
cursorMark = [0, 0]

# Field settings
field_width_cm = 220
margin = 20
arucoRadius = 5
corner = 0.5

# Initialize Planner (20x20 grid) with Dijkstra
planner = DijkstraPlanner(field_width_cm, field_width_cm, grid_size=20)
current_path = []


def aruco_iden(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            if markerID == 5: Field_ArMark_Center[0] = [cX, cY]
            if markerID == 3: Field_ArMark_Center[1] = [cX, cY]
            if markerID == 0: Field_ArMark_Center[2] = [cX, cY]
            if markerID == 4: Field_ArMark_Center[3] = [cX, cY]
            if markerID == 1: RobotMarker[0] = [cX, cY]

            if markerID == 2:
                ObjectMarkers[0] = [cX, cY]
            if markerID == 6:
                ObjectMarkers[1] = [cX, cY]
            if markerID == 7:
                ObjectMarkers[2] = [cX, cY]
            if markerID == 8:
                ObjectMarkers[3] = [cX, cY]
            if markerID == 9:
                ObjectMarkers[4] = [cX, cY]


def on_mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if 'pix2cmX' in globals():
            cursorMark[0] = x / pix2cmX
            cursorMark[1] = y / pix2cmY
            print(f"Goal Set: {cursorMark}")


def getTransformMatrix():
    pts1 = np.float32([Field_ArMark_Center[2], Field_ArMark_Center[3], Field_ArMark_Center[0], Field_ArMark_Center[1]])
    pts2 = np.float32([(margin, margin), (margin, margin + field_width_cm * 2.5), (margin + field_width_cm * 2.5, margin), (margin + field_width_cm * 2.5, margin + field_width_cm * 2.5)])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    return matrix


def draw_radii(img):
    """Draw the radii circles for robot and objects"""
    h, w, _ = img.shape

    # Draw robot radius (blue circle)
    if RobotPosition[0] != (0, 0):
        rx, ry = RobotPosition[0]
        rx_pix = int(rx * pix2cmX)
        ry_pix = int(ry * pix2cmY)
        robot_radius_pix = int(ROBOT_RADIUS * pix2cmX)

        # Draw robot radius circle (semi-transparent)
        overlay = img.copy()
        cv.circle(overlay, (rx_pix, ry_pix), robot_radius_pix, (255, 0, 0), 2)  # Blue outline
        cv.circle(overlay, (rx_pix, ry_pix), robot_radius_pix, (255, 100, 100), -1)  # Filled with transparency
        cv.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Draw robot center
        cv.circle(img, (rx_pix, ry_pix), 5, (255, 0, 0), -1)

        # Add text label
        cv.putText(img, f"Robot R={ROBOT_RADIUS}cm", (rx_pix + 15, ry_pix - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw object radii (red circles)
    for i, obs in enumerate(ObjectPosition):
        if obs[0] != 0 and obs[1] != 0:
            ox_pix = int(obs[0] * pix2cmX)
            oy_pix = int(obs[1] * pix2cmY)
            object_radius_pix = int(OBJECT_RADIUS * pix2cmX)

            # Draw object radius circle (semi-transparent)
            overlay = img.copy()
            cv.circle(overlay, (ox_pix, oy_pix), object_radius_pix, (0, 0, 255), 2)  # Red outline
            cv.circle(overlay, (ox_pix, oy_pix), object_radius_pix, (100, 100, 255), -1)  # Filled with transparency
            cv.addWeighted(overlay, 0.2, img, 0.8, 0, img)

            # Draw object center
            cv.circle(img, (ox_pix, oy_pix), 4, (0, 0, 255), -1)

            # Add text label
            cv.putText(img, f"O{i + 1} R={OBJECT_RADIUS}cm", (ox_pix + 10, oy_pix - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


def draw_grid_and_path(img, path, grid_size=20):
    h, w, _ = img.shape
    step_x = w / grid_size
    step_y = h / grid_size

    # Draw Grid Lines (lighter)
    for i in range(grid_size + 1):
        cv.line(img, (int(i * step_x), 0), (int(i * step_x), h), (100, 100, 100), 1)
        cv.line(img, (0, int(i * step_y)), (w, int(i * step_y)), (100, 100, 100), 1)

    # Draw Obstacle-affected cells (darker red to show the safety margin effect)
    for obs in ObjectPosition:
        if obs[0] != 0 and obs[1] != 0:
            # Mark all cells within safety margin
            margin = OBJECT_RADIUS + ROBOT_RADIUS
            for i in range(grid_size):
                for j in range(grid_size):
                    cell_center = planner.grid_to_world((i, j))
                    dx = cell_center[0] - obs[0]
                    dy = cell_center[1] - obs[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= margin:
                        top_left = (int(i * step_x), int(j * step_y))
                        bot_right = (int((i + 1) * step_x), int((j + 1) * step_y))
                        # Different shades based on distance
                        if dist <= OBJECT_RADIUS:
                            # Object core area (dark red)
                            cv.rectangle(img, top_left, bot_right, (0, 0, 150), -1)
                        else:
                            # Safety margin area (lighter red)
                            cv.rectangle(img, top_left, bot_right, (50, 50, 150), -1)

    # Draw Path (Green Lines)
    if len(path) > 1:
        for i in range(len(path) - 1):
            p1 = (int(path[i][0] * pix2cmX), int(path[i][1] * pix2cmY))
            p2 = (int(path[i + 1][0] * pix2cmX), int(path[i + 1][1] * pix2cmY))
            cv.line(img, p1, p2, (0, 255, 0), 3)
            cv.circle(img, p1, 5, (0, 255, 0), -1)

        # Draw target point
        if len(path) > 0:
            target = (int(path[-1][0] * pix2cmX), int(path[-1][1] * pix2cmY))
            cv.circle(img, target, 8, (0, 255, 255), -1)
            cv.putText(img, "GOAL", (target[0] + 10, target[1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_legend(img):
    """Draw legend explaining the colors and radii"""
    h, w, _ = img.shape
    y_start = 30
    x_start = 20

    # Background for legend
    cv.rectangle(img, (10, 10), (280, 150), (0, 0, 0), -1)
    cv.rectangle(img, (10, 10), (280, 150), (255, 255, 255), 1)

    # Algorithm name
    cv.putText(img, "Dijkstra's Algorithm", (x_start, y_start - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Robot
    cv.circle(img, (x_start + 10, y_start + 20), 8, (255, 0, 0), -1)
    cv.putText(img, f"Robot (R={ROBOT_RADIUS}cm)", (x_start + 25, y_start + 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Object
    cv.circle(img, (x_start + 10, y_start + 45), 6, (0, 0, 255), -1)
    cv.putText(img, f"Object (R={OBJECT_RADIUS}cm)", (x_start + 25, y_start + 50),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Path
    cv.line(img, (x_start + 5, y_start + 70), (x_start + 20, y_start + 70), (0, 255, 0), 2)
    cv.putText(img, "Planned Path", (x_start + 25, y_start + 75),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Safety margin
    cv.rectangle(img, (x_start + 5, y_start + 95), (x_start + 20, y_start + 105), (50, 50, 150), -1)
    cv.putText(img, "Safety Margin", (x_start + 25, y_start + 102),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Diagonal movement indicator
    cv.putText(img, "8-direction movement", (x_start, y_start + 125),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


def liveCameraWork():
    unwarped_frame = cv.warpPerspective(frame, matrix, resTrack)
    unwarped_frame = unwarped_frame[0:600, 0:600]

    corners, id, rejected = cv.aruco.detectMarkers(unwarped_frame, dictionary)
    aruco_iden(corners, id, rejected, unwarped_frame)
    cv.aruco.drawDetectedMarkers(unwarped_frame, corners, id)

    # Update Positions
    RobotPosition[0] = (RobotMarker[0][0] / pix2cmX, RobotMarker[0][1] / pix2cmY)
    for x in range(5):
        ObjectPosition[x] = (ObjectMarkers[x][0] / pix2cmX, ObjectMarkers[x][1] / pix2cmY)

    # --- DIJKSTRA PLANNING LOGIC WITH RADII ---
    global current_path
    if cursorMark[0] != 0:
        current_path = planner.plan(start_pos=RobotPosition[0], target_pos=cursorMark, obstacles=ObjectPosition, object_radius=OBJECT_RADIUS, robot_radius=ROBOT_RADIUS)

    # Draw everything
    draw_grid_and_path(unwarped_frame, current_path)
    draw_radii(unwarped_frame)
    # draw_legend(unwarped_frame)

    # Draw cursor position
    if cursorMark[0] != 0:
        cursor_pix = (int(cursorMark[0] * pix2cmX), int(cursorMark[1] * pix2cmY))
        cv.circle(unwarped_frame, cursor_pix, 5, (255, 255, 0), -1)

    # Display algorithm info
    cv.putText(unwarped_frame, "Using: Dijkstra's Algorithm", (10, 580),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv.imshow('Laboratory Work', unwarped_frame)


# --- SETUP ---
ret, frame = vid.read()
time.sleep(0.5)
ret, frame = vid.read()

corners, id, rejected = cv.aruco.detectMarkers(frame, dictionary)
aruco_iden(corners, id, rejected, frame)
matrix = getTransformMatrix()

unwarped_frame = cv.warpPerspective(frame, matrix, resTrack)[0:600, 0:600]
aruco_iden(*cv.aruco.detectMarkers(unwarped_frame, dictionary) + (unwarped_frame,))

pix2cmX = abs(Field_ArMark_Center[0][0] - Field_ArMark_Center[2][0]) / (field_width_cm - 2 * corner - 2 * arucoRadius)
pix2cmY = abs(Field_ArMark_Center[0][1] - Field_ArMark_Center[1][1]) / (field_width_cm - 2 * corner - 2 * arucoRadius)
cv.imshow('Laboratory Work', unwarped_frame)
cv.setMouseCallback('Laboratory Work', on_mouse_click)
sock = connect_to_robotino()

print("System Ready. Click to set target.")
print(f"Using: Dijkstra's Algorithm")
print(f"Robot Radius: {ROBOT_RADIUS}cm, Object Radius: {OBJECT_RADIUS}cm")
print(f"8-direction movement with diagonal costs")

# --- MAIN LOOP ---
while True:
    ret, frame = vid.read()
    if not ret: break

    liveCameraWork()

    frame_counter += 1
    # Build target position from cursorMark (convert to tuple)
    target = (cursorMark[0], cursorMark[1]) if cursorMark[0] != 0 or cursorMark[1] != 0 else None
    log_frame_data(csv_writer, frame_counter, RobotPosition[0], target, current_path)

    # --- CONTROL LOGIC (unchanged) ---
    if cursorMark[0] != 0 and len(current_path) > 1:
        target_wp = current_path[1]
        rx, ry = RobotPosition[0]
        tx, ty = target_wp

        vx = (tx - rx) * 0.8
        vy = (ty - ry) * 0.8

        speed = math.sqrt(vx ** 2 + vy ** 2)
        max_speed = 10
        if speed > max_speed:
            scale = max_speed / speed
            vx *= scale
            vy *= scale

        print(f"Path len: {len(current_path)} | Moving to Grid: {target_wp}")
        send_velocity(-vx / 100.0, vy / 100.0, 0)

    elif cursorMark[0] != 0 and len(current_path) <= 1:
        print("Target Reached or No Path Found")
        send_velocity(0, 0, 0)
    else:
        send_velocity(0, 0, 0)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()