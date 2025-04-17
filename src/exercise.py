# filepath: /Users/71-Sota-Nakahara/code/Herniated Disc Rehab Suite/src/exercise.py
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
import yaml

# MediaPipeとAbslのログレベルを設定
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


@dataclass
class TimerConfig:
    """
    タイマー設定を保持するデータクラス。
    """

    enabled: bool
    duration: int = 0
    repetitions: int = 1
    rest_time: int = 0


@dataclass
class AngleRule:
    """
    角度のルールを定義するデータクラス。
    """

    min_angle: float
    max_angle: float
    description: str


@dataclass
class Condition:
    """
    複数の角度ルールを組み合わせた条件を定義するデータクラス。
    """

    type: str  # "AND" または "OR"
    rules: List[AngleRule]


@dataclass
class AngleSet:
    """
    計算する角度のセットと、その条件を定義するデータクラス。
    """

    landmarks: List[str]
    conditions: List[Condition]


@dataclass
class Stretch:
    """
    ストレッチの情報を保持するデータクラス。
    """

    name: str
    key: int
    angle_sets: List[AngleSet]
    timer: TimerConfig


class StretchExercise:
    """
    ストレッチエクササイズを管理するクラス。
    """

    def __init__(self, config_file: str):
        """
        StretchExerciseを初期化します。

        Parameters:
            config_file (str): 設定ファイル（YAML）のパス。
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
        self.start_sound_played = False
        self.end_sound_played = False
        self.grace_start: Optional[float] = None
        # タイマー開始状態を追跡するフラグを追加
        self._timer_started = False

    # タイマー開始状態を確認するプロパティを追加
    @property
    def timer_started(self) -> bool:
        """タイマーが開始されているかどうか"""
        return self._timer_started or self.timer_start is not None

    # タイマーを開始するメソッドを追加
    def start_timer(self):
        """タイマーを開始"""
        if not self._timer_started and self.current_stretch and self.current_stretch.timer.enabled:
            self.timer_start = time.time()
            self._timer_started = True
            self.is_resting = False
            self.start_sound_played = False
            self.end_sound_played = False
            print(f"Timer started for {self.current_stretch.name}")
            return True
        return False

    def _load_config(self, config_file: str) -> Dict[int, Stretch]:
        """
        設定ファイルを読み込み、ストレッチの設定をロードします。

        Parameters:
            config_file (str): 設定ファイル（YAML）のパス。

        Returns:
            Dict[int, Stretch]: ストレッチのキーとStretchオブジェクトの辞書。
        """
        try:
            # ファイルパスの存在チェックを追加
            if not os.path.exists(config_file):
                print(f"Config file not found: {config_file}")
                # デフォルト設定を返す
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
                if key > 0:  # 有効なキーかチェック
                    stretches[key] = Stretch(name=stretch_config.get("name", f"Stretch {key}"), key=key, angle_sets=angle_sets, timer=timer)

            return stretches

        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            return {}

    def calculate_angle(self, a, b, c):
        """
        3点からなる角度を計算します。

        Parameters:
            a, b, c: 各点の座標。

        Returns:
            float: 計算された角度。
        """
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)

            ba = a - b
            bc = c - b

            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)

            # ゼロ除算を防ぐ
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
        角度が指定された条件を満たすかチェックします。

        Parameters:
            angle (float): チェックする角度。
            condition (Condition): 判定条件。

        Returns:
            bool: 条件を満たす場合はTrue。
        """
        if condition.type == "AND":
            return all(rule.min_angle <= angle <= rule.max_angle for rule in condition.rules)
        else:
            return any(rule.min_angle <= angle <= rule.max_angle for rule in condition.rules)

    def play_sound(self, sound_file):
        """
        サウンドファイルを非同期で再生します。

        Parameters:
            sound_file (str): 再生するサウンドファイルのパス。
        """
        if os.path.exists(sound_file):
            Thread(target=playsound3.playsound, args=(sound_file,), daemon=True).start()
        else:
            print(f"Warning: Sound file not found: {sound_file}")

    def draw_text(self, image, text, x, y, font_scale=0.7, color=(255, 255, 255), thickness=2):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def process_frame(self, frame, pose, landmarks, return_angles=False):
        """
        フレームを処理し、ストレッチの状態を更新します。

        Parameters:
            frame: カメラからのフレーム。
            pose: MediaPipeのポーズ推定オブジェクト。
            landmarks: ポーズランドマーク。
            return_angles: 角度情報を返すかどうか

        Returns:
            image: 処理後のフレーム。
            angle_status: 各関節の角度と評価結果 (return_angles=Trueの場合)
        """
        # ストレッチが選択されていない場合は基本的なランドマーク表示のみ
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

        # 角度評価結果を格納する辞書
        angle_status = {}

        # 半透明の黒いオーバーレイを追加してテキスト読みやすくする
        h, w = image.shape[:2]
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)  # 上部に黒いバー
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)  # 透明度30%で合成

        # 現在のストレッチ名を画面上部に表示
        stretch_name = self.current_stretch.name
        if isinstance(stretch_name, str):
            try:
                stretch_name = stretch_name.encode("utf-8").decode("utf-8")
            except UnicodeError:
                stretch_name = stretch_name.encode("ascii", errors="ignore").decode()
        self.draw_text(image, f"ストレッチ: {stretch_name}", 10, 30, font_scale=0.9)

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

            # 各条件を個別に判定
            conditions_met = []
            for condition in angle_set.conditions:
                condition_met = self.check_stretch_condition(angle, condition)
                conditions_met.append(condition_met)
                for rule in condition.rules:
                    if rule.description:
                        all_descriptions.append(rule.description)

            # 角度セット全体として条件を満たすか判定
            set_conditions_met = any(conditions_met)  # OR条件に変更（どれか1つの条件が満たされればOK）
            all_conditions_met = all_conditions_met and set_conditions_met

            # 角度情報と評価結果を記録
            angle_status[joint_name] = {
                "angle": angle,
                "is_correct": set_conditions_met,
                "description": ", ".join([rule.description for cond in angle_set.conditions for rule in cond.rules if rule.description]),
            }

            # 角度と結果を角度ごとに分かりやすく表示
            base_y = 80 + angle_set_index * 40

            # 角度値と条件の結果を表示する部分を修正
            angle_text = f"角度 {angle_set_index+1}: {int(angle)}°"
            self.draw_text(image, angle_text, 10, base_y, font_scale=0.8)

            # OK/NGの表示位置を計算
            text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            x_offset = 10 + text_size[0] + 10  # 角度テキストの後ろに少し空白を入れる

            # 判定結果を表示
            result_text = "OK" if set_conditions_met else "NG"
            color_txt = (0, 255, 0) if set_conditions_met else (0, 0, 255)
            self.draw_text(image, result_text, x_offset, base_y, font_scale=0.8, color=color_txt, thickness=2)

            # 角度ラインの描画
            cv2.line(image, points[0], points[1], (255, 255, 0), 3)  # 1つ目の線
            cv2.line(image, points[1], points[2], (255, 255, 0), 3)  # 2つ目の線

            # 各点を色付きで強調表示
            for i, point in enumerate(points):
                color = (0, 0, 255) if i == 1 else (0, 255, 255)  # 中心点は赤、その他は黄色
                cv2.circle(image, point, 8, color, -1)

            # 角度の弧を描画
            angle_viz_radius = 30

            # 角度を可視化する円弧を描く
            ba = np.array(points[0]) - np.array(points[1])
            bc = np.array(points[2]) - np.array(points[1])

            start_angle = np.arctan2(ba[1], ba[0])
            end_angle = np.arctan2(bc[1], bc[0])

            # 円弧の開始角と終了角を調整
            if start_angle > end_angle:
                start_angle, end_angle = end_angle, start_angle

            # 円弧の色を条件に応じて変更
            arc_color = (0, 255, 0) if set_conditions_met else (0, 0, 255)

            # 円弧を描画（OpenCVは時計回りに角度を測定するため調整が必要）
            cv2.ellipse(image, points[1], (angle_viz_radius, angle_viz_radius), 0, np.degrees(start_angle), np.degrees(end_angle), arc_color, 2)

            # 角度値をテキストで表示
            angle_text_pos = (
                int(points[1][0] + angle_viz_radius * 1.5 * np.cos((start_angle + end_angle) / 2)),
                int(points[1][1] + angle_viz_radius * 1.5 * np.sin((start_angle + end_angle) / 2)),
            )
            cv2.putText(image, f"{int(angle)}°", angle_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, arc_color, 2)

        # タイマー状態の更新
        self.update_timer(all_conditions_met)

        # フィードバック領域のオーバーレイ
        base_y = max(80 + len(self.angles) * 40, 180)
        text_bg_height = min(base_y + 120, h - 20)  # 画面の下部を超えないように
        overlay = image.copy()
        cv2.rectangle(overlay, (0, base_y - 40), (350, text_bg_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

        # タイマー情報の表示
        if self.current_stretch.timer.enabled:
            timer_y = base_y
            if self.is_resting:
                rest_remaining = self.current_stretch.timer.rest_time - (time.time() - self.rest_start)
                self.draw_text(image, f"休憩時間: {int(rest_remaining)} 秒", 10, timer_y, font_scale=0.9, color=(0, 165, 255))
            elif self.timer_start is not None:
                elapsed = time.time() - self.timer_start
                remaining = max(0, self.current_stretch.timer.duration - elapsed)
                self.draw_text(image, f"残り時間: {int(remaining)} 秒", 10, timer_y, font_scale=0.9, color=(0, 255, 255))
            self.draw_text(
                image, f"セット: {self.current_rep}/{self.current_stretch.timer.repetitions}", 10, timer_y + 40, font_scale=0.9, color=(255, 255, 255)
            )

        # 指示テキストを表示
        for i, desc in enumerate(all_descriptions):
            desc_y = text_bg_height - 20 - (len(all_descriptions) - 1 - i) * 30
            self.draw_text(image, f"• {desc}", 10, desc_y, font_scale=0.7, color=(255, 255, 255))

        # 姿勢のOK/NGに基づいて全体的な色を設定
        pose_color = (0, 255, 0) if all_conditions_met else (0, 0, 255)

        # 姿勢の正確さに基づいて枠を描画
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), pose_color, 3)

        # 角度情報を返すかどうか
        return (image, angle_status) if return_angles else image

    def set_stretch(self, key: int):
        """
        現在のストレッチを設定します。

        Parameters:
            key (int): 設定するストレッチのキー。

        Returns:
            bool: ストレッチの設定に成功した場合はTrue。
        """
        if key in self.stretches:
            self.current_stretch = self.stretches[key]
            self.timer_start = None
            self._timer_started = False  # タイマー開始フラグをリセット
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
        """タイマーの状態を更新します"""
        if not self.current_stretch or not self.current_stretch.timer.enabled:
            return

        current_time = time.time()

        if all_conditions_met and not self.is_resting:
            if self.timer_start is None:
                # タイマーがまだ開始されていない場合
                if not self._timer_started:
                    self.timer_start = current_time
                    if not self.start_sound_played:
                        self.play_sound(os.path.join("sounds", "start.wav"))
                        self.start_sound_played = True
                    self.end_sound_played = False
                    self._timer_started = True  # タイマー開始フラグを設定
                    print(f"Timer started at {self.timer_start}")
            # timer_start が設定されている場合のみ経過時間を計算
            if self.timer_start is not None:
                elapsed_time = current_time - self.timer_start
                if elapsed_time >= self.current_stretch.timer.duration:
                    self._handle_timer_completion()
        elif self.is_resting:
            self._handle_rest_time()
        else:
            self._handle_grace_period()

    def _handle_timer_completion(self):
        """タイマー完了時の処理を行います"""
        if self.current_rep < self.current_stretch.timer.repetitions:
            if not self.end_sound_played:
                self.play_sound(os.path.join("sounds", "end.wav"))
                self.end_sound_played = True
                print(f"Completed rep {self.current_rep} of {self.current_stretch.timer.repetitions}")
            self.current_rep += 1
            self.is_resting = True
            self.rest_start = time.time()
            print(f"Rest started at {self.rest_start}, rest time: {self.current_stretch.timer.rest_time}s")
        else:
            if not self.end_sound_played:
                self.play_sound(os.path.join("sounds", "allend.wav"))
                self.end_sound_played = True
                print("All reps completed!")
            self.current_rep = 1
            self._timer_started = False  # タイマー完了時にフラグを落とす
        self.timer_start = None
        self.start_sound_played = False
        self.grace_start = None

    def _handle_rest_time(self):
        """休憩時間の処理を行います"""
        rest_elapsed = time.time() - self.rest_start
        if rest_elapsed >= self.current_stretch.timer.rest_time:
            self.is_resting = False
            self.rest_start = None
            self.start_sound_played = False
            self.end_sound_played = False

    def _handle_grace_period(self):
        """猶予期間の処理を行います"""
        if self.grace_start is None:
            self.grace_start = time.time()
        elif time.time() - self.grace_start >= 2:
            self.timer_start = None
            self.start_sound_played = False
            self.end_sound_played = False
            self.grace_start = None
            self._timer_started = False  # タイマー開始フラグをリセット

    @staticmethod
    def init_camera():
        """カメラを初期化して返します"""
        for i in range(2):  # まず0番、次に1番を試す
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = 640
                height = 480
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap, width, height
        return None, None, None

    def draw_menu(self, frame):
        """メニュー画面を描画します"""
        self.draw_text(frame, "press the key", 50, 50, font_scale=1)
        self.draw_text(frame, "1-9: select the stretch", 50, 100)
        self.draw_text(frame, "q: quit", 50, 150)
        self.draw_text(frame, "space: back to home", 50, 200)

        y = 250
        for key_num, stretch in sorted(self.stretches.items()):
            self.draw_text(frame, f"{key_num}: {stretch.name}", 50, y)
            y += 30


def main():
    """
    メイン関数。カメラを起動し、ストレッチエクササイズを開始します。
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
    # 実行ブロックをコメントアウトし、ライブラリとして利用されるようにします。
    # main()
    pass
