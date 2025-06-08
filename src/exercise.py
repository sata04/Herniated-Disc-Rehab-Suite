# -*- coding: utf-8 -*-
import logging
import os  # for handling file paths
import time
from dataclasses import dataclass
from threading import Thread
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
import playsound3
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import yaml

# MediaPipeとAbslのログレベルを設定
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


@dataclass
class TimerConfig:
    """
    Data class to hold timer settings.
    """

    enabled: bool
    duration: int = 0
    repetitions: int = 1
    rest_time: int = 0


@dataclass
class AngleRule:
    """
    Data class to define angle rules.
    """

    min_angle: float
    max_angle: float
    description: str


@dataclass
class Condition:
    """
    Data class to define conditions that combine multiple angle rules.
    """

    type: str  # "AND" or "OR"
    rules: List[AngleRule]


@dataclass
class AngleSet:
    """
    Data class to define sets of angles to calculate and their conditions.
    """

    landmarks: List[str]
    conditions: List[Condition]


@dataclass
class Stretch:
    """
    Data class to hold stretch information.
    """

    name: str
    key: int
    angle_sets: List[AngleSet]
    timer: TimerConfig


class StretchExercise:
    """
    Class to manage stretch exercises.
    """

    def __init__(self, config_file: str):
        """
        Initializes StretchExercise.

        Parameters:
            config_file (str): Path to the configuration file (YAML).
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.stretches = self._load_config(config_file)
        self.current_stretch: Optional[Stretch] = None
        self.timer_start: Optional[float] = None
        self.current_rep = 1
        self.is_resting = False
        self.rest_start: Optional[float] = None
        self.angles: List[float] = []
        self.grace_start: Optional[float] = None
        # Add a flag to track the timer start state
        self._timer_started = False
        self.sound_to_play: Optional[str] = None

    # Add a property to check the timer start state
    @property
    def timer_started(self) -> bool:
        """Whether the timer has started"""
        return self._timer_started or self.timer_start is not None

    # Add a method to start the timer
    def start_timer(self):
        """Start the timer"""
        if not self._timer_started and self.current_stretch and self.current_stretch.timer.enabled:
            self.timer_start = time.time()
            self._timer_started = True
            self.is_resting = False
            print(f"Timer started for {self.current_stretch.name}")
            return True
        return False

    def stop_timer(self):
        """Stop the timer"""
        self.timer_start = None
        self._timer_started = False
        self.grace_start = None
        print("Timer stopped")
        return True

    def _load_config(self, config_file: str) -> Dict[int, Stretch]:
        """
        Reads the configuration file and loads stretch settings.

        Parameters:
            config_file (str): Path to the configuration file (YAML).

        Returns:
            Dict[int, Stretch]: Dictionary of stretch keys and Stretch objects.
        """
        try:
            # Add a check for file path existence
            if not os.path.exists(config_file):
                print(f"Config file not found: {config_file}")
                # Return default settings
                return {
                    1: Stretch(
                        name="Default Stretch",
                        key=1,
                        angle_sets=[
                            AngleSet(
                                landmarks=["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
                                conditions=[Condition(type="AND", rules=[AngleRule(min_angle=150, max_angle=180, description="Default position")])],
                            )
                        ],
                        timer=TimerConfig(enabled=True, duration=10, repetitions=1, rest_time=5),
                    )
                }

            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            stretches = {}
            stretch_list = config.get("stretches", [])

            if not stretch_list:
                print("No stretches found in config file")
                return {}

            for stretch_config in stretch_list:
                timer_config = stretch_config.get("timer", {})
                timer = TimerConfig(
                    enabled=timer_config.get("enabled", False),
                    duration=timer_config.get("duration", 0),
                    repetitions=timer_config.get("repetitions", 1),
                    rest_time=timer_config.get("rest_time", 0),
                )

                angle_sets = []
                for angle_set in stretch_config.get("angle_sets", []):
                    conditions = []
                    for condition_config in angle_set.get("conditions", []):
                        rules = []
                        for rule in condition_config.get("rules", []):
                            rules.append(AngleRule(min_angle=rule["angle"][0], max_angle=rule["angle"][1], description=rule.get("description", "")))
                        conditions.append(Condition(type=condition_config.get("type", "AND"), rules=rules))

                    angle_sets.append(AngleSet(landmarks=angle_set["landmarks"], conditions=conditions))

                key = int(stretch_config.get("key", 0))
                if key > 0:  # Check if the key is valid
                    stretches[key] = Stretch(name=stretch_config.get("name", f"Stretch {key}"), key=key, angle_sets=angle_sets, timer=timer)

            return stretches

        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            return {}

    def calculate_angle(self, a, b, c):
        """
        Calculates the angle formed by three points.

        Parameters:
            a, b, c: Coordinates of each point.

        Returns:
            float: Calculated angle.
        """
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)

            ba = a - b
            bc = c - b

            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)

            # Prevent division by zero
            if norm_ba < 1e-6 or norm_bc < 1e-6:
                return 0.0

            cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angle = np.degrees(angle)

            return angle

        except Exception as e:
            print(f"Angle calculation error: {e}")
            return 0.0

    def check_stretch_condition(self, angle: float, condition: Condition) -> bool:
        """
        Checks if the angle meets the specified condition.

        Parameters:
            angle (float): Angle to check.
            condition (Condition): Judgment condition.

        Returns:
            bool: True if the condition is met.
        """
        if condition.type == "AND":
            return all(rule.min_angle <= angle <= rule.max_angle for rule in condition.rules)
        else:
            return any(rule.min_angle <= angle <= rule.max_angle for rule in condition.rules)

    def draw_text(self, image, text, x, y, font_scale=0.7, color=(255, 255, 255), thickness=2):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def process_frame(self, frame, pose, landmarks, return_angles=False):
        """
        Processes the frame and updates the stretch state.

        Parameters:
            frame: Frame from the camera.
            pose: MediaPipe pose estimation object.
            landmarks: Pose landmarks.
            return_angles: Whether to return angle information.

        Returns:
            image: Processed frame.
            angle_status: Angle and evaluation result for each joint (if return_angles=True).
        """
        # If no stretch is selected, display basic landmarks only
        if self.current_stretch is None:
            image = frame.copy()
            self.mp_drawing.draw_landmarks(
                image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            )
            return (image, {}) if return_angles else image

        image = frame.copy()

        self.angles = []
        all_conditions_met = True
        all_descriptions = []

        # Dictionary to store angle evaluation results
        angle_status = {}

        # Add a translucent black overlay to make text easier to read
        h, w = image.shape[:2]
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)  # Black bar at the top
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)  # Composite with 30% transparency

        # Display the current stretch name at the top of the screen
        stretch_name = self.current_stretch.name
        if isinstance(stretch_name, str):
            try:
                stretch_name = stretch_name.encode("utf-8").decode("utf-8")
            except UnicodeError:
                stretch_name = stretch_name.encode("ascii", errors="ignore").decode()
        self.draw_text(image, f"Stretch: {stretch_name}", 10, 30, font_scale=0.9)

        for angle_set_index, angle_set in enumerate(self.current_stretch.angle_sets):
            points = []
            point_indices = []
            joint_name = f"angle_{angle_set_index+1}"

            for landmark_name in angle_set.landmarks:
                landmark_index = getattr(self.mp_pose.PoseLandmark, landmark_name).value
                landmark = landmarks.landmark[landmark_index]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                points.append((x, y))
                point_indices.append(landmark_index)

            angle = self.calculate_angle(points[0], points[1], points[2])
            self.angles.append(angle)

            # Judge each condition individually
            conditions_met = []
            for condition in angle_set.conditions:
                condition_met = self.check_stretch_condition(angle, condition)
                conditions_met.append(condition_met)
                for rule in condition.rules:
                    if rule.description:
                        all_descriptions.append(rule.description)

            # Judge if the angle set as a whole meets the conditions
            set_conditions_met = any(conditions_met)  # Change to OR condition (OK if any one condition is met)
            all_conditions_met = all_conditions_met and set_conditions_met

            # Record angle information and evaluation results
            angle_status[joint_name] = {
                "angle": angle,
                "is_correct": set_conditions_met,
                "description": ", ".join([rule.description for cond in angle_set.conditions for rule in cond.rules if rule.description]),
            }

            # 角度と結果を角度ごとに分かりやすく表示
            base_y = 80 + angle_set_index * 40

            # Correct the part that displays angle values and condition results
            angle_text = f"Angle {angle_set_index+1}: {int(angle)}°"
            self.draw_text(image, angle_text, 10, base_y, font_scale=0.8)

            # Calculate the display position of OK/NG
            text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            x_offset = 10 + text_size[0] + 10  # Add a little space after the angle text

            # Display judgment result
            result_text = "OK" if set_conditions_met else "NG"
            color_txt = (0, 255, 0) if set_conditions_met else (0, 0, 255)
            self.draw_text(image, result_text, x_offset, base_y, font_scale=0.8, color=color_txt, thickness=2)

            # Draw angle lines
            cv2.line(image, points[0], points[1], (255, 255, 0), 3)  # First line
            cv2.line(image, points[1], points[2], (255, 255, 0), 3)  # Second line

            # Highlight each point with color
            for i, point in enumerate(points):
                color = (0, 0, 255) if i == 1 else (0, 255, 255)  # Center point is red, others are yellow
                cv2.circle(image, point, 8, color, -1)

            # Draw the arc of the angle
            angle_viz_radius = 30

            # Draw an arc to visualize the angle - improved calculation method
            ba = np.array(points[0]) - np.array(points[1])
            bc = np.array(points[2]) - np.array(points[1])

            # Calculate angle (using dot product)
            # cosine_angle is not needed because it is generated directly from the angle for arc drawing

            # Determine the start and end angles of the arc considering the vector direction
            start_angle = np.arctan2(ba[1], ba[0])
            end_angle = np.arctan2(bc[1], bc[0])

            # Set the angle range of the arc appropriately
            # Corresponds to cases where the angle exceeds 180 degrees
            if abs(end_angle - start_angle) > np.pi:
                if end_angle > start_angle:
                    start_angle += 2 * np.pi
                else:
                    end_angle += 2 * np.pi

            # Change the arc color according to the condition
            arc_color = (0, 255, 0) if set_conditions_met else (0, 0, 255)

            # OpenCV's ellipse requires angles in degrees
            start_deg = np.degrees(start_angle)
            end_deg = np.degrees(end_angle)

            # Adjust start and end angles to ensure the shorter arc is drawn
            if abs(end_deg - start_deg) > 180:
                if end_deg > start_deg:
                    end_deg, start_deg = start_deg, end_deg

            # Draw arc
            cv2.ellipse(image, points[1], (angle_viz_radius, angle_viz_radius), 0, start_deg, end_deg, arc_color, 2)

            # Display angle value as text - adjust position
            mid_angle = (start_angle + end_angle) / 2
            angle_text_pos = (
                int(points[1][0] + angle_viz_radius * 1.8 * np.cos(mid_angle)),
                int(points[1][1] + angle_viz_radius * 1.8 * np.sin(mid_angle)),
            )
            cv2.putText(image, f"{int(angle)}°", angle_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, arc_color, 2)

        # Update timer status
        # Not called here if return_angles=True because timer control is handled by VideoTransformer
        if not return_angles:
            self.update_timer(all_conditions_met)

        # Overlay for feedback area
        base_y = max(80 + len(self.angles) * 40, 180)
        text_bg_height = min(base_y + 120, h - 20)  # Do not exceed the bottom of the screen
        overlay = image.copy()
        cv2.rectangle(overlay, (0, base_y - 40), (350, text_bg_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

        # Display timer information (English)
        if self.current_stretch.timer.enabled:
            timer_y = base_y  # Y coordinate for timer display
            if self.is_resting:
                rest_remaining = max(0, self.current_stretch.timer.rest_time - (time.time() - self.rest_start))
                self.draw_text(image, f"Rest time: {int(rest_remaining)} sec", 10, timer_y, font_scale=0.9, color=(0, 165, 255))
            elif self.timer_start is not None:
                elapsed = time.time() - self.timer_start
                remaining = max(0, self.current_stretch.timer.duration - elapsed)
                self.draw_text(image, f"Time left: {int(remaining)} sec", 10, timer_y, font_scale=0.9, color=(0, 255, 255))
            self.draw_text(
                image, f"Set: {self.current_rep}/{self.current_stretch.timer.repetitions}", 10, timer_y + 40, font_scale=0.9, color=(255, 255, 255)
            )

        # Display instruction text
        for i, desc in enumerate(all_descriptions):
            desc_y = text_bg_height - 20 - (len(all_descriptions) - 1 - i) * 30
            self.draw_text(image, f"• {desc}", 10, desc_y, font_scale=0.7, color=(255, 255, 255))

        # Set overall color based on posture OK/NG
        pose_color = (0, 255, 0) if all_conditions_met else (0, 0, 255)

        # Display status text
        status_text = "CORRECT" if all_conditions_met else "INCORRECT"  # Text for posture status (CORRECT/INCORRECT)
        status_color = (0, 255, 0) if all_conditions_met else (0, 0, 255)

        # Display status text prominently at the top of the screen
        cv2.rectangle(image, (w - 200, 5), (w - 20, 45), (0, 0, 0), -1)
        self.draw_text(image, status_text, w - 190, 35, font_scale=0.8, color=status_color, thickness=2)

        # Draw a border based on posture accuracy
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), pose_color, 3)

        # Whether to return angle information
        return (image, angle_status) if return_angles else image

    def set_stretch(self, key: int):
        """
        Sets the current stretch.

        Parameters:
            key (int): Key of the stretch to set.

        Returns:
            bool: True if setting the stretch was successful.
        """
        if key in self.stretches:
            self.current_stretch = self.stretches[key]
            self.timer_start = None
            self._timer_started = False  # Reset timer start flag
            self.current_rep = 1
            self.is_resting = False
            self.rest_start = None
            self.angles = []
            print(f"Changed stretch to: {self.current_stretch.name}")
            return True
        else:
            print("Invalid stretch number")
        return False

    def update_timer(self, all_conditions_met):
        """Updates the timer state"""
        self.sound_to_play = None # Reset sound indicator at the beginning of each update cycle
        if not self.current_stretch or not self.current_stretch.timer.enabled:
            return

        current_time = time.time()

        if all_conditions_met and not self.is_resting:
            # Processing when posture is correct and not resting
            if self.timer_start is None:
                # If the timer has not started yet
                if not self._timer_started:
                    # Ensure timer start processing
                    self.timer_start = current_time
                    self.sound_to_play = "start"
                    self._timer_started = True  # Set timer start flag
                    self.grace_start = None  # Reset grace period
                    print(f"Timer started at {self.timer_start}")
            # Calculate elapsed time only if timer_start is set
            if self.timer_start is not None:
                elapsed_time = current_time - self.timer_start
                if elapsed_time >= self.current_stretch.timer.duration:
                    self._handle_timer_completion()
        elif self.is_resting:
            self._handle_rest_time()
        else:
            # Processing when posture is incorrect
            self._handle_grace_period(current_time)

    def _handle_timer_completion(self):
        """Handles timer completion"""
        if self.current_rep < self.current_stretch.timer.repetitions:
            self.sound_to_play = "end"
            print(f"Completed rep {self.current_rep} of {self.current_stretch.timer.repetitions}")
            self.current_rep += 1
            self.is_resting = True
            self.rest_start = time.time()
            print(f"Rest started at {self.rest_start}, rest time: {self.current_stretch.timer.rest_time}s")
        else:
            self.sound_to_play = "allend"
            print("All reps completed!")
            self.current_rep = 1
            self._timer_started = False  # Clear flag when timer is complete
        self.timer_start = None
        self.grace_start = None

    def _handle_rest_time(self):
        """Handles rest time"""
        rest_elapsed = time.time() - self.rest_start
        if rest_elapsed >= self.current_stretch.timer.rest_time:
            self.is_resting = False
            self.rest_start = None

    def _handle_grace_period(self, current_time=None):
        """Handles grace period"""
        if current_time is None:
            current_time = time.time()

        if self.grace_start is None:
            self.grace_start = current_time
        elif current_time - self.grace_start >= 2:
            self.timer_start = None
            self.grace_start = None
            self._timer_started = False  # Reset timer start flag

    @staticmethod
    def init_camera():
        """Initializes and returns the camera"""
        for i in range(2):  # Try 0 first, then 1
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = 640
                height = 480
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap, width, height
        return None, None, None

    def draw_menu(self, frame):
        """Draws the menu screen"""
        self.draw_text(frame, "Press the key", 50, 50, font_scale=1)
        self.draw_text(frame, "1-9: Select the stretch", 50, 100)
        self.draw_text(frame, "Q: Quit", 50, 150)
        self.draw_text(frame, "Space: Back to home", 50, 200)

        y = 250
        for key_num, stretch in sorted(self.stretches.items()):
            self.draw_text(frame, f"{key_num}: {stretch.name}", 50, y)
            y += 30


def main():
    """
    Main function. Starts the camera and begins the stretch exercise.
    """
    exercise = StretchExercise(os.path.join("config", "stretch_config.yaml"))
    cap, width, height = exercise.init_camera()
    if cap is None:
        print("Failed to open camera")
        return

    cv2.namedWindow("Stretch Exercise", cv2.WINDOW_NORMAL)
    current_mode = "menu"
    prev_frame_time = 0

    with exercise.mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1,
    ) as pose:
        while cap.isOpened():
            if current_mode == "menu":
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                exercise.draw_menu(frame)
                cv2.imshow("Stretch Exercise", frame)
            elif current_mode == "exercise":
                ret, frame = cap.read()
                if not ret:
                    print("Failed to get frame")
                    break

                frame = cv2.flip(frame, 1)

                frame.flags.writeable = False
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame.flags.writeable = True

                if results.pose_landmarks:
                    frame = exercise.process_frame(frame, pose, results.pose_landmarks)

                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                cv2.putText(
                    frame,
                    f"FPS: {int(fps)}",
                    (frame.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Stretch Exercise", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                exercise.current_stretch = None
                current_mode = "menu"
            elif ord("1") <= key <= ord("9"):
                if exercise.set_stretch(key - ord("0")):
                    current_mode = "exercise"
                    print(f"Changed stretch to: {exercise.current_stretch.name}")
                else:
                    print("Invalid stretch number")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Comment out the execution block to use as a library.
    # main()
    pass
