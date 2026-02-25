import os
import numpy as np

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import time
import math
from Robotino_communication import connect_to_robotino, get_odometry, get_proximity_sensor_values, send_velocity

vid = cv.VideoCapture(0, cv.CAP_MSMF)  # 2 sec
# vid = cv.VideoCapture('FieldRecognition/AllIn.mp4')
#
resTrack = (1600, 1200)
vid.set(cv.CAP_PROP_FRAME_WIDTH, resTrack[0])
vid.set(cv.CAP_PROP_FRAME_HEIGHT, resTrack[1])
vid.set(cv.CAP_PROP_FPS, 30)

dict = cv.aruco.DICT_5X5_100
dictionary = cv.aruco.getPredefinedDictionary(dict)

prev_frame_time = 0
new_frame_time = 0

# Corner 1, 2, 3, 4 [x, y]
Field_ArMark_Center = [(0, 0), (0, 0), (0, 0), (0, 0)]  # Id 0, 3, 4, 5

# Object aruco markers center position
ObjectMarkers = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]  # Id 2, 6, 7, 8, 9
ObjectPosition = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]  # Id 2, 6, 7, 8, 9

# Robot marker
RobotMarker = [(0, 0)]  # Id 1
RobotPosition = [(0, 0)]  # Id 1
RobotsMarkers = [(0, 0)]


# Coordinates of the button press
cursorMark = [0, 0]

# Field measurements
field = 220
corner = 0.5
arucoRadius = 5
margin = 20

# Proportional coefficient for the speeds
Kp = 0.5
Krepelling = 3.2

# Vx, vy
V = [0, 0]
V_repel = [0, 0]
V_final = [0, 0]

# Obstacle avoidance parameters
OBSTACLE_INFLUENCE_RADIUS = 35  # cm - distance at which obstacles start affecting robot
OBSTACLE_REPULSION_GAIN = 3  # gain factor for repulsion force
ROBOT_FIELD_RADIUS = 45  # cm - robot's personal space


def distance_two_points(x1, x2, y1, y2):
    d = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
    return d


def aruco_iden(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            if markerID == 5:
                Field_ArMark_Center[0] = [cX, cY]
            if markerID == 3:
                Field_ArMark_Center[1] = [cX, cY]
            if markerID == 0:
                Field_ArMark_Center[2] = [cX, cY]
            if markerID == 4:
                Field_ArMark_Center[3] = [cX, cY]
            if markerID == 1:
                RobotMarker[0] = [cX, cY]
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
        cursorMark[0] = x / pix2cmX
        cursorMark[1] = y / pix2cmY
        print(cursorMark)


def arucoDraw(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv.putText(image, str(markerID), (cX, cY), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 2)
            # print("[Inference] ArUco marker ID: {}".format(markerID))

    return image


def getTransformMatrix():
    pts1 = np.float32([Field_ArMark_Center[2], Field_ArMark_Center[3], Field_ArMark_Center[0], Field_ArMark_Center[1]])
    pts2 = np.float32([(margin, margin), (margin, margin + field * 2.5), (margin + field * 2.5, margin), (margin + field * 2.5, margin + field * 2.5)])

    matrix = cv.getPerspectiveTransform(pts1, pts2)
    # print(matrix)
    return matrix


def perspective_transform_sustained(frame, matrix):
    field = cv.warpPerspective(frame, matrix, resTrack)
    return field


def liveCameraWork():
    # Fix frame with matrix
    unwarped_frame = perspective_transform_sustained(frame, matrix)
    #  Crop frame so that only the field was visible
    unwarped_frame = unwarped_frame[0:600, 0:600]
    # Detect markers on the fixed field
    corners, id, rejected = cv.aruco.detectMarkers(unwarped_frame, dictionary)
    #  post process to find the center of each or the fields' markers
    aruco_iden(corners, id, rejected, unwarped_frame)
    # Draw the centers and outline of each aruco marker
    unwarped_frame = arucoDraw(corners, id, rejected, unwarped_frame)

    # cv.circle(unwarped_frame, (RobotMarker[0][0], RobotMarker[0][1]), 5, (0, 0, 255), -1)
    cv.circle(unwarped_frame, (RobotMarker[0][0], RobotMarker[0][1]), int(ROBOT_FIELD_RADIUS * pix2cmY), (0, 0, 255), 2)

    RobotPosition[0] = (RobotMarker[0][0] / pix2cmX, RobotMarker[0][1] / pix2cmY)
    # Calculate the aruco markers' position in the field
    for x in range(0, 5):
        ObjectPosition[x] = (ObjectMarkers[x][0] / pix2cmX, ObjectMarkers[x][1] / pix2cmY)
        cv.circle(unwarped_frame, (int(ObjectMarkers[x][0]), int(ObjectMarkers[x][1])), int(OBSTACLE_INFLUENCE_RADIUS * pix2cmY), (0, 0, 255), 2)

    # Speed line (purple)
    cv.arrowedLine(unwarped_frame, (RobotMarker[0][0], RobotMarker[0][1]), (int(RobotMarker[0][0] + 10*V[0]), int(RobotMarker[0][1] + 10*V[1])), (150, 0, 150), 2)
    # repelling line (red)
    cv.arrowedLine(unwarped_frame, (RobotMarker[0][0], RobotMarker[0][1]), (int(RobotMarker[0][0] + 10 * V_repel[0]), int(RobotMarker[0][1] + 10 * V_repel[1])), (0, 0, 250), 2)
    # Final speed line (blue)
    cv.arrowedLine(unwarped_frame, (RobotMarker[0][0], RobotMarker[0][1]), (int(RobotMarker[0][0] + 10 * V_final[0]), int(RobotMarker[0][1] + 10 * V_final[1])), (250, 0, 0), 2)
    return unwarped_frame


def obstacle_avoidance_force_field(robot_pos, obstacles_pos):
    vx_repel = 0
    vy_repel = 0

    robot_x, robot_y = robot_pos[0]

    for obs_pos in obstacles_pos:
        # Skip if obstacle detection failed
        if obs_pos[0] == 0 and obs_pos[1] == 0:
            continue

        obs_x, obs_y = obs_pos

        # 1. Calculate vector from Obstacle TO Robot
        dx = robot_x - obs_x
        dy = robot_y - obs_y

        # 2. Calculate real Euclidean distance
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # Avoid division by zero if robot is ON TOP of obstacle
        if dist < 0.1:
            dist = 0.1

            # 3. Check if inside influence radius
        if dist < ROBOT_FIELD_RADIUS + OBSTACLE_INFLUENCE_RADIUS:
            # 4. Calculate Repulsion Magnitude (Inverse Square Law)
            # We use (dist/100) to keep units consistent with your previous tuning
            force_magnitude = Krepelling / ((dist / 100) ** 2)

            # 5. Project force into X and Y components (Unit Vector * Magnitude)
            vx_repel += (dx / dist) * force_magnitude
            vy_repel += (dy / dist) * force_magnitude

    return vx_repel, vy_repel


# Get first frame
ret, frame = vid.read()

# cv.imshow('w', frame)
time.sleep(0.2)
# detect aruco markers
corners, id, rejected = cv.aruco.detectMarkers(frame, dictionary)
#  post process data from aruco markers
aruco_iden(corners, id, rejected, frame)
#  Print field orienting aruco markers center
print("Field Aruco markers")
print(Field_ArMark_Center)
#  Calculate matrix
matrix = getTransformMatrix()
# Fix frame with matrix
unwarped_frame = perspective_transform_sustained(frame, matrix)
#  Crop frame so that only the field was visible
unwarped_frame = unwarped_frame[0:600, 0:600]
# Show the unwarped frame
cv.imshow('Fixed field', unwarped_frame)

# Detect markers on the fixed field
corners, id, rejected = cv.aruco.detectMarkers(unwarped_frame, dictionary)
#  post process to find the center of each or the fields' markers
aruco_iden(corners, id, rejected, unwarped_frame)
# Draw the centers and outline of each aruco marker
unwarped_frame = arucoDraw(corners, id, rejected, unwarped_frame)
#  Show the unwarped frame with aruco markers
cv.imshow('Laboratory Work', unwarped_frame)
# Add mouse clicking capabilities to the window
cv.setMouseCallback('Laboratory Work', on_mouse_click)
# Print the new center of each corners' marker
print("Fixed field aruco markers")
print(Field_ArMark_Center)
#  calculate the pix2cm in the X axis
pix2cmX = abs(Field_ArMark_Center[0][0] - Field_ArMark_Center[2][0]) / (field - 2 * corner - 2 * arucoRadius)
#  calculate the pix2cm in the X axis
pix2cmY = abs(Field_ArMark_Center[0][1] - Field_ArMark_Center[1][1]) / (field - 2 * corner - 2 * arucoRadius)
# Print the centers of the aruco markers that ARENT the field
print("Objects' markers")
print(ObjectMarkers)
print("Robots marker")
print(RobotMarker)

# Calculate the aruco markers' position in the field
for x in range(0, 5):
    ObjectPosition[x] = (ObjectMarkers[x][0] / pix2cmX, ObjectMarkers[x][1] / pix2cmY)
print("Objects' position")
print(ObjectPosition)
# Calculate the robots position
RobotPosition[0] = (RobotMarker[0][0] / pix2cmX, RobotMarker[0][1] / pix2cmY)
print("Robot position")
print(RobotPosition)
initialRobotPos = (RobotMarker[0][0], RobotMarker[0][1] )
# Connect to robotino
sock = connect_to_robotino()

# Saving the initial position of the robots odometry
# odometry_readings = get_odometry()
# x_0 = odometry_readings[0]
# y_0 = odometry_readings[1]

while True:
    ret, frame = vid.read()
    # new_frame_time = time.time()
    # fps = 1 / (new_frame_time - prev_frame_time)
    # prev_frame_time = new_frame_time
    # print(fps)
    final_frame = liveCameraWork()
    #  Show the unwarped frame with aruco markers
    # Showing the trajectory of the robot
    RobotsMarkers.append((RobotMarker[0][0], RobotMarker[0][1]))
    for (x, y) in RobotsMarkers:
        cv.circle(final_frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Showing the starting and finishing point of the robot
    cv.circle(final_frame, (int(initialRobotPos[0]), int(initialRobotPos[1])), 5, (0, 255, 255), 2)
    cv.circle(final_frame, (int(cursorMark[0]*pix2cmX), int(cursorMark[1]*pix2cmY)), 5, (255, 0, 255), 2)

    cv.imshow('Laboratory Work', final_frame)
    # Only if we selected a speed on the GUI with the mouse then the robot moves
    if cursorMark[0] != 0:
        speedLimit = 10
        deadZone = 0.1

        # Calculate attraction force towards target
        vx = (cursorMark[0] - RobotPosition[0][0]) * Kp
        if abs(vx) < deadZone:
            vx = 0
        elif abs(vx) > speedLimit:
            vx = speedLimit * abs(vx) / vx

        vy = (cursorMark[1] - RobotPosition[0][1]) * Kp
        if abs(vy) < deadZone:
            vy = 0
        elif abs(vy) > speedLimit:
            vy = speedLimit * abs(vy) / vy

        print(f"Initial: ({vx:.2f}, {vy:.2f})")
        V = (vx, vy)
        # Calculate repulsion force from obstacles
        vx_repel, vy_repel = obstacle_avoidance_force_field(RobotPosition, ObjectPosition)
        V_repel = (vx_repel, vy_repel)
        # Add repulsion to attraction (with scaling factor to balance forces)
        vx += vx_repel * 1
        vy += vy_repel * 1

        # Apply speed limits again after adding repulsion
        if abs(vx) > speedLimit:
            vx = speedLimit * abs(vx) / vx
        if abs(vy) > speedLimit:
            vy = speedLimit * abs(vy) / vy
        V_final = (vx, vy)
        print("Speed setpoints (with obstacle avoidance)")
        print(f"Repulsion: ({vx_repel:.2f}, {vy_repel:.2f})")
        print(f"Final: ({vx:.2f}, {vy:.2f})")
        print("Cursor")
        print(cursorMark)
        print("Robots position")
        print(RobotPosition)
        send_velocity(-vx / 100, vy / 100, 0)
    else:
        send_velocity(0, 0, 0)

    time.sleep(0.1)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()