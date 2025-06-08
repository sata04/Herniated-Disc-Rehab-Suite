# -*- coding: utf-8 -*-
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import av
import av.logging
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.exercise import StretchExercise

# TensorFlow/MediaPipeの警告抑制のための環境変数設定
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlowのログを非表示
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # GPUに関する警告を抑制
os.environ["AUTOGRAPH_VERBOSITY"] = "0"  # AutoGraphの冗長出力を抑制
os.environ["TF_CPP_VMODULE"] = "xnnpack_delegate=0"  # XNNPACKデレゲートのログを抑制

# Configure log levels to suppress verbose output
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)  # MediaPipeのログを抑制
av.logging.set_level(av.logging.ERROR)

# MediaPipeの特定の警告を無視するフィルター追加
mediapipe_logger = logging.getLogger("mediapipe")
mediapipe_logger.setLevel(logging.ERROR)
mediapipe_logger.addFilter(
    lambda record: False if "landmark_projection_calculator" in record.getMessage() or "feedback manager" in record.getMessage().lower() else True
)

# TensorFlowの警告を完全に無視
tf_logger = logging.getLogger("tensorflow")
tf_logger.setLevel(logging.ERROR)  # ERRORよりも厳格なFATAL

# Streamlitの特定の警告を無視
warnings.filterwarnings("ignore", message="missing ScriptRunContext")
warnings.filterwarnings("ignore", category=DeprecationWarning)  # 非推奨警告を無視

# Page configuration
st.set_page_config(
    page_title="Herniated Disc Rehab Suite",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "# Herniated Disc Rehab Suite\nAn interactive rehabilitation app for herniated disc patients."},
)

# Initialize session state variables
if "current_key" not in st.session_state:
    st.session_state.current_key = None
if "exercise_start_time" not in st.session_state:
    st.session_state.exercise_start_time = None
if "current_rep" not in st.session_state:
    st.session_state.current_rep = 0
if "exercise_phase" not in st.session_state:
    st.session_state.exercise_phase = "idle"  # can be "idle", "active", "rest"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "pending_sound" not in st.session_state:
    st.session_state.pending_sound = None

# Initialize posture correctness state
if "posture_ok" not in st.session_state:
    st.session_state.posture_ok = False

# Sidebar UI with improved styling
st.sidebar.title("Herniated Disc Rehab Suite")
st.sidebar.markdown("---")

# Sidebar App Settings
with st.sidebar.expander("⚙️ App Settings", expanded=False):
    # ダークモード設定が変更されたかを追跡するために古い値を保存
    # old_dark_mode = st.session_state.dark_mode # No longer needed for localStorage
    st.checkbox("Dark Mode", value=st.session_state.dark_mode, key="dark_mode", help="Toggle dark/light mode")

    # ダークモード設定が変更された場合、ローカルストレージに保存 # Removed localStorage logic
    # if old_dark_mode != st.session_state.dark_mode:
    #     st.markdown(
    #         f"""
    #         <script>
    #         // ダークモード設定をローカルストレージに保存
    #         localStorage.setItem('dark_mode', '{str(st.session_state.dark_mode).lower()}');
    #         </script>
    #         """,
    #         unsafe_allow_html=True,
    #     )

    # 角度表示オプションを追加
    st.checkbox("Show Joint Angles", value=st.session_state.get("show_angles", False), key="show_angles", help="Display joint angle measurements")

# 解像度とモデル複雑度の設定
st.sidebar.markdown("---")
# Local variables for resolution and model complexity
resolution = st.sidebar.selectbox("Camera Resolution", ["640x480", "1280x720"], help="Choose camera resolution")
model_complexity = st.sidebar.selectbox("Pose Model Complexity", [0, 1, 2], index=1, help="Set MediaPipe model complexity")

# Load stretch configuration
config_path = Path(__file__).parent / "config" / "stretch_config.yaml"

# Load stretch options
exercise = StretchExercise(str(config_path))
stretch_keys = sorted(exercise.stretches.keys())

# Create a more visually appealing exercise selector
st.sidebar.subheader("Select Exercise")
selected_key = st.sidebar.selectbox(
    "Choose a stretch exercise",
    options=stretch_keys,
    format_func=lambda k: exercise.stretches[k].name,
    help="Select the rehabilitation exercise you want to perform",
)

# Add exercise control buttons with better styling
col1, col2 = st.sidebar.columns(2)
start = col1.button("▶️ Start", use_container_width=True)
stop = col2.button("⏹️ Stop", use_container_width=True)

# Exercise control logic
if start:
    st.session_state.current_key = selected_key
    st.session_state.exercise_start_time = None  # will be set when internal timer starts
    st.session_state.current_rep = 1
    st.session_state.exercise_phase = "active"
elif stop:
    st.session_state.current_key = None
    st.session_state.exercise_start_time = None
    st.session_state.current_rep = 0
    st.session_state.exercise_phase = "idle"

# Display current date and time in sidebar footer
st.sidebar.markdown("---")
current_date = datetime.now().strftime("%Y-%m-%d")
st.sidebar.caption(f"Today: {current_date}")

# Main layout
st.title("💪 Stretch Exercise Rehabilitation")

# Display different content based on exercise state
if st.session_state.current_key is None:
    # Welcome screen with instructions
    st.markdown(
        """
    <div class="welcome-card">
        <h2>Welcome to Your Rehabilitation Journey</h2>
        <p>This application helps you perform guided exercises for herniated disc rehabilitation.</p>
        <ol>
            <li>Select an exercise from the sidebar</li>
            <li>Click <b>Start</b> to begin your session</li>
            <li>Follow the on-screen guidance and posture feedback</li>
            <li>Complete all repetitions for maximum benefit</li>
        </ol>
        <p>💡 <i>Tip: Ensure your entire body is visible in the camera frame for best results.</i></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display sample exercises in a grid
    st.subheader("Available Exercises")
    exercise_cols = st.columns(3)
    for i, (key, stretch) in enumerate(exercise.stretches.items()):
        with exercise_cols[i % 3]:
            st.markdown(
                f"""
            <div class="exercise-card">
                <h3>{stretch.name}</h3>
                <p><b>Duration:</b> {str(stretch.timer.duration) if stretch.timer.enabled else 'N/A'} seconds</p>
                <p><b>Repetitions:</b> {str(stretch.timer.repetitions) if stretch.timer.enabled else 'N/A'}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
else:
    # Active exercise view with improved layout
    # Create a 70/30 split layout for video and controls
    video_col, info_col = st.columns([7, 3])

    with info_col:
        # Get current exercise details
        stretch = exercise.stretches[st.session_state.current_key]

        # Create styled exercise info card
        st.markdown(
            f"""
        <div class="info-card">
            <h2>{stretch.name}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Display timer and progress if available
        if stretch.timer.enabled:
            st.markdown("### Exercise Progress")

            # Calculate and display progress
            if st.session_state.exercise_start_time is not None:
                # Calculate time variables
                elapsed = time.time() - st.session_state.exercise_start_time
                current_rep = st.session_state.current_rep
                total_reps = stretch.timer.repetitions

                # Display repetition counter
                st.markdown(
                    f"""
                <div class="counter">
                    <span class="counter-value">{current_rep}</span>
                    <span class="counter-label">/ {total_reps} reps</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Calculate and show phase (exercise or rest)
                cycle_time = stretch.timer.duration + stretch.timer.rest_time
                position_in_cycle = elapsed % cycle_time

                if position_in_cycle < stretch.timer.duration:
                    # In exercise phase
                    st.session_state.exercise_phase = "active"
                    phase_progress = position_in_cycle / stretch.timer.duration
                    remaining = stretch.timer.duration - position_in_cycle
                    st.markdown("<p class='phase active-phase'>Exercise Phase</p>", unsafe_allow_html=True)
                else:
                    # In rest phase
                    st.session_state.exercise_phase = "rest"
                    phase_progress = (position_in_cycle - stretch.timer.duration) / stretch.timer.rest_time
                    remaining = cycle_time - position_in_cycle
                    st.markdown("<p class='phase rest-phase'>Rest Phase</p>", unsafe_allow_html=True)

                # Display progress bar
                st.progress(phase_progress)

                # Display countdown
                mins, secs = divmod(int(remaining), 60)
                st.markdown(
                    f"""
                <div class="timer">
                    <span class="time-value">{mins:02d}:{secs:02d}</span>
                    <span class="time-label">remaining</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Display exercise details
            st.markdown("### Details")
            st.markdown(
                f"""
            <div class="details-card">
                <div class="detail-item">
                    <span class="detail-label">Duration:</span>
                    <span class="detail-value">{str(stretch.timer.duration)} sec</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Repetitions:</span>
                    <span class="detail-value">{str(stretch.timer.repetitions)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Rest Period:</span>
                    <span class="detail-value">{str(stretch.timer.rest_time)} sec</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No timer configured for this exercise. Perform at your own pace.")

    with video_col:
        # Video transformer for live webcam processing
        # Parse resolution into width and height for media constraints
        width, height = map(int, resolution.split("x"))

        class VideoTransformer(VideoProcessorBase):
            def __init__(self):
                # 初期化処理を軽量化
                # Use the global config_path defined outside this class
                try:
                    self.exercise = StretchExercise(str(config_path)) # Use global config_path
                    current_key = st.session_state.get("current_key", None)
                    if current_key is not None:
                        success = self.exercise.set_stretch(current_key)
                        if not success:
                            print(f"Failed to set stretch key: {current_key}")
                    else:
                        print("No current stretch key set")
                except Exception as e:
                    print(f"Error initializing StretchExercise: {e}")
                    self.exercise = None

                # MediaPipe Poseモデルの初期化
                self.pose = mp.solutions.pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=1,  # デフォルト値を使用
                    static_image_mode=False,  # 動画ストリーム向けトラッキング
                )

                # 描画ユーティリティの参照を追加
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_pose = mp.solutions.pose

                # State for phase and FPS calculation
                self.phase = st.session_state.get("exercise_phase", "idle")
                self.last_frame_time = time.time()

                # 評価状態を追加
                self.angle_status = {}
                self.is_pose_correct = False

                # フィードバック状態
                self.feedback_text = "Detecting posture..."
                self.feedback_color = (255, 255, 255)
                self.hold_timer = None  # Noneで未計測を明示
                self.hold_required = 2.0  # 正しい姿勢を維持する必要がある秒数
                self.grace_start = None
                self.grace_period = 2.0  # 猶予期間（秒）

            def _handle_grace_period(self, current_time):
                """猶予期間の処理ロジック"""
                if self.grace_start is None:
                    self.grace_start = current_time
                elif current_time - self.grace_start > self.grace_period:
                    # 猶予期間経過でタイマーリセット
                    if self.exercise and self.exercise.timer_started:
                        self.exercise.stop_timer()
                        self.hold_timer = None
                    self.feedback_text = "Incorrect posture! Please adjust."
                    self.feedback_color = (0, 0, 255)  # 赤色

            def recv(self, frame):
                """フレームを受信して処理"""
                try:
                    img = frame.to_ndarray(format="bgr24")
                    # フリップ操作を追加して自然な向きにする
                    img = cv2.flip(img, 1)

                    # 画面サイズを取得
                    h, w = img.shape[:2]

                    # MediaPipe処理用にRGB変換
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_rgb.flags.writeable = False

                    # MediaPipeによるポーズ検出処理
                    results = self.pose.process(img_rgb)

                    # 描画用に戻す
                    img_rgb.flags.writeable = True
                    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    # 状態と時間の更新
                    current_time = time.time()
                    fps = 1.0 / max(0.001, current_time - self.last_frame_time)
                    self.last_frame_time = current_time

                    # デバッグモードの指定
                    debug_mode = st.session_state.get("show_angles", False)

                    # 描画のベースレイヤーをコピー
                    display_img = img.copy()

                    # ポーズ検出をデバッグ表示
                    cv2.putText(display_img, f"FPS: {int(fps)}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # ポーズが検出されたかどうか確認
                    if results.pose_landmarks:
                        # ポーズが検出されたときの処理
                        if self.exercise is not None and st.session_state.get("current_key") is not None:
                            try:
                                # StretchExerciseクラスを使って姿勢評価
                                processed_img, self.angle_status = self.exercise.process_frame(
                                    img, self.pose, results.pose_landmarks, return_angles=True
                                )

                                # 処理が成功したら、処理された画像を使用
                                display_img = processed_img

                                # 角度評価結果からポーズの正確さを判定
                                conditions_met = [status.get("is_correct", False) for status in self.angle_status.values()]
                                self.is_pose_correct = all(conditions_met) if conditions_met else False

                                # タイマー未開始時: 正しい姿勢を一定時間保持でタイマー開始
                                if st.session_state.get("exercise_phase") == "active" and not self.exercise.timer_started:
                                    if self.is_pose_correct:
                                        self.grace_start = None
                                        if self.hold_timer is None:
                                            self.hold_timer = current_time
                                            self.feedback_text = f"Hold posture... {self.hold_required:.1f}s"
                                            self.feedback_color = (0, 255, 255)
                                        else:
                                            hold_duration = current_time - self.hold_timer
                                            if hold_duration >= self.hold_required:
                                                success = self.exercise.start_timer()
                                                self.hold_timer = None
                                                self.feedback_text = "Exercise started! Timer running."
                                                self.feedback_color = (0, 255, 0)
                                                if success:
                                                    st.session_state.exercise_start_time = current_time
                                            else:
                                                remaining = self.hold_required - hold_duration
                                                self.feedback_text = f"Hold posture... {remaining:.1f}s"
                                                self.feedback_color = (0, 255, 255)
                                    else:
                                        self.hold_timer = None
                                        self._handle_grace_period(current_time)
                                # タイマー動作中
                                elif st.session_state.get("exercise_phase") == "active" and self.exercise.timer_started:
                                    self.exercise.update_timer(self.is_pose_correct)
                                    if not self.is_pose_correct:
                                        self._handle_grace_period(current_time)
                                    else:
                                        self.grace_start = None
                                        self.feedback_text = "Exercising... Maintain posture."
                                        self.feedback_color = (0, 255, 0)
                                    if self.exercise.is_resting:
                                        self.feedback_text = "Resting... Prepare for the next set."
                                        self.feedback_color = (0, 165, 255)

                                # 各関節の角度表示（デバッグ情報）
                                if debug_mode:
                                    y_pos = 100
                                    for joint, data in self.angle_status.items():
                                        if "angle" in data:
                                            status_color = (0, 255, 0) if data.get("is_correct", False) else (0, 0, 255)
                                            angle_text = f"{joint}: {data['angle']:.1f}° {'✓' if data.get('is_correct', False) else '✗'}"
                                            cv2.putText(display_img, angle_text, (w - 250, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                                            y_pos += 25
                                            # ルールの説明を表示
                                            if data.get("description"):
                                                cv2.putText(
                                                    display_img,
                                                    data["description"],
                                                    (w - 250, y_pos),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5,
                                                    (255, 255, 255),
                                                    1,
                                                )
                                                y_pos += 25

                                # 姿勢ステータス（OK/NG）表示を強調表示
                                posture_text = "Posture OK" if self.is_pose_correct else "Posture NG"
                                posture_color = (0, 255, 0) if self.is_pose_correct else (0, 0, 255)
                                # 濃い背景で目立たせる
                                cv2.rectangle(display_img, (20, 15), (200, 45), (0, 0, 0), -1)
                                cv2.putText(display_img, posture_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, posture_color, 3)

                                # フィードバックテキスト表示を追加
                                # テキスト背景を追加
                                feedback_size = cv2.getTextSize(self.feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                text_w = feedback_size[0][0]
                                cv2.rectangle(display_img, (30, h - 50), (30 + text_w + 20, h - 20), (0, 0, 0), -1)
                                cv2.putText(display_img, self.feedback_text, (40, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.feedback_color, 2)

                                # セッション状態を更新
                                st.session_state.current_rep = self.exercise.current_rep
                                st.session_state.exercise_phase = "rest" if self.exercise.is_resting else "active"
                                st.session_state.posture_ok = self.is_pose_correct

                                # Check for sound events from StretchExercise
                                if self.exercise.sound_to_play:
                                    st.session_state.pending_sound = self.exercise.sound_to_play
                                    self.exercise.sound_to_play = None  # Reset immediately

                            except Exception as e:
                                print(f"Error processing frame: {str(e)}")
                                # エラーメッセージを画面に表示
                                error_msg = f"Error: {str(e)[:50]}"
                                cv2.putText(display_img, error_msg, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # ストレッチが選択されていない場合、基本的なポーズ表示のみ
                            self.mp_drawing.draw_landmarks(
                                display_img,
                                results.pose_landmarks,
                                self.mp_pose.POSE_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                            )

                            cv2.putText(
                                display_img, "Please select a stretch", (w // 2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                            )
                    else:
                        # ポーズが検出されない場合のプロンプト
                        cv2.putText(display_img, "Pose not detected", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(
                            display_img, "Please stand in front of the camera", (w // 2 - 170, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                        )

                    return av.VideoFrame.from_ndarray(display_img, format="bgr24")

                except Exception as e:
                    print(f"Fatal error in recv method: {str(e)}")
                    # エラーが発生した場合でも何かを返す必要がある
                    error_frame = frame.to_ndarray(format="bgr24")
                    cv2.putText(error_frame, f"Error: {str(e)[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return av.VideoFrame.from_ndarray(error_frame, format="bgr24")

        # Improved webcam streamer UI with better framing
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        webrtc_streamer(
            key=f"stretch-exercise-{st.session_state.current_key}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoTransformer,
            media_stream_constraints={
                "video": {"width": {"ideal": width}, "height": {"ideal": height}, "facingMode": "user"},  # フロントカメラを優先、自然な向きを保持
                "audio": False,
            },
            async_processing=True,
            video_html_attrs={
                "style": {"width": "100%", "height": "auto", "margin": "0 auto", "display": "block", "transform": "scaleX(1)"},
                "controls": False,
                "autoPlay": True,
                "muted": True,
            },
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Tips and guidance under the video
        with st.expander("Exercise Tips & Guidance", expanded=True):
            st.markdown(
                """
            - Keep your back straight during the exercise
            - Breathe slowly and deeply throughout the movement
            - Stop immediately if you feel sharp pain
            - Focus on quality of movement, not speed
            """
            )

        # ポーズ正誤ステータス表示
        if st.session_state.get("posture_ok", False) is not None:
            status = st.session_state.posture_ok
            # フィードバックコンテナを使ってOK/NGを表示
            container_class = "feedback-correct" if status else "feedback-incorrect"
            label = "OK" if status else "NG"
            html = f"<div class='feedback-container {container_class}'>" f"<h3>姿勢: {label}</h3></div>"
            st.markdown(html, unsafe_allow_html=True)


# サウンドファイルの存在確認用ヘルパー関数
def safe_audio(file_path):
    """指定された音声ファイルが存在すればそれを再生、なければ無音ファイルを生成して再生"""
    # 相対パスを絶対パスに変換
    base_dir = Path(__file__).parent
    sound_path = base_dir / file_path

    # 音声ファイルが存在するか確認
    if sound_path.exists():
        return st.audio(str(sound_path))
    else:
        # 音声ファイルが見つからない場合の処理
        st.warning(f"Sound file not found: {sound_path}")
        # デバッグ用に検索パスを表示
        logging.info(f"Sound file search path: {sound_path}")
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Base directory: {base_dir}")

        # 無音ファイルのフォールバックディレクトリを作成
        fallback_dir = base_dir / "sounds/fallback"
        fallback_dir.mkdir(exist_ok=True, parents=True)
        fallback_file = fallback_dir / "silent.wav"

        # 無音ファイルが存在しない場合は作成
        try:
            import numpy as np
            import scipy.io.wavfile as wav

            # 1秒間の無音を生成（サンプルレート44100、16bitステレオ）
            sample_rate = 44100
            data = np.zeros((sample_rate, 2), dtype=np.int16)
            wav.write(str(fallback_file), sample_rate, data)
        except ImportError:
            # scipyがない場合は警告のみ
            st.error("Failed to generate sound file. Please install the scipy library.")
            return None
        except Exception as e:
            st.error(f"An error occurred while generating the sound file: {str(e)}")
            return None

        # 無音ファイルを再生
        return st.audio(str(fallback_file))


# サウンドフィードバック機能の強化とリアルタイム通知
if st.session_state.current_key is not None:
    # Play sound if pending
    if st.session_state.get("pending_sound"):
        sound_file_key = st.session_state.pending_sound
        if sound_file_key in ["start", "end", "allend"]: # Ensure only valid keys are used
            safe_audio(f"sounds/{sound_file_key}.wav")
        st.session_state.pending_sound = None  # Clear the sound cue


# Apply custom CSS for modern look
st.markdown(
    f"""
<style>
/* Base styles with dark mode support */
:root {{
    --primary-color: {("#2e7bcf", "#4da6ff")[st.session_state.dark_mode]};
    --bg-color: {("#f5f7fa", "#1e1e1e")[st.session_state.dark_mode]};
    --card-bg: {("#ffffff", "#2d2d2d")[st.session_state.dark_mode]};
    --text-color: {("#333333", "#e0e0e0")[st.session_state.dark_mode]};
    --border-color: {("#e0e0e0", "#555555")[st.session_state.dark_mode]};
    --shadow-color: {("rgba(0,0,0,0.1)", "rgba(0,0,0,0.3)")[st.session_state.dark_mode]};
    --active-color: #4CAF50;
    --rest-color: #FF9800;
    --feedback-color: #2196F3;
}}

/* Main container styling */
.stApp {{
    background-color: var(--bg-color);
    color: var(--text-color);
    transition: all 0.3s ease;
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
    background: {("linear-gradient(180deg, #2e7bcf, #2a62b1)", "linear-gradient(180deg, #2d2d2d, #1a1a1a)")[st.session_state.dark_mode]};
    color: white;
}}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: white !important;
}}

/* Card styling */
.welcome-card, .exercise-card, .info-card, .details-card {{
    background-color: var(--card-bg);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px var(--shadow-color);
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}}

.welcome-card:hover, .exercise-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 6px 12px var(--shadow-color);
}}

.exercise-card {{
    height: 180px;
    overflow: hidden;
}}

.exercise-card h3 {{
    color: var(--primary-color);
    margin-top: 0;
}}

.details-card {{
    padding: 15px;
}}

.detail-item {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 5px;
}}

.detail-label {{
    font-weight: bold;
    color: var(--primary-color);
}}

/* Header styling */
h1, h2, h3 {{
    color: var(--primary-color) !important;
    font-weight: 600;
}}

/* Video container styling */
.video-container {{
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 6px 18px var(--shadow-color);
    margin-bottom: 20px;
}}

video {{
    border-radius: 8px;
    max-width: 100%;
}}

/* Progress indicators */
.counter, .timer {{
    text-align: center;
    margin: 15px 0;
}}

.counter-value, .time-value {{
    font-size: 2.5rem;
    font-weight: bold;
    color: var(--primary-color);
}}

.counter-label, .time-label {{
    font-size: 1rem;
    color: var(--text-color);
    opacity: 0.8;
}}

/* Phase indicator */
.phase {{
    text-align: center;
    padding: 8px;
    border-radius: 8px;
    font-weight: bold;
    margin: 10px 0;
}}

.active-phase {{
    background-color: var(--active-color);
    color: white;
}}

.rest-phase {{
    background-color: var(--rest-color);
    color: white;
}}

/* Button styling */
button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}}

button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 8px var(--shadow-color);
}}

/* Expander styling */
.streamlit-expanderHeader {{
    font-weight: 600;
    color: var(--primary-color);
}}

/* Exercise ID styling */
.exercise-id {{
    background-color: {("rgba(46, 123, 207, 0.1)", "rgba(77, 166, 255, 0.2)")[st.session_state.dark_mode]};
    color: var(--primary-color);
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-top: -5px;
}}

/* Responsive design adjustments */
@media screen and (max-width: 992px) {{
    .counter-value, .time-value {{
        font-size: 2rem;
    }}

    .exercise-card {{
        height: auto;
    }}
}}

/* 評価フィードバック表示のスタイル */
.feedback-container {{
    background-color: var(--card-bg);
    border-left: 4px solid var(--feedback-color);
    padding: 10px 15px;
    margin: 10px 0;
    border-radius: 4px;
    box-shadow: 0 2px 4px var(--shadow-color);
}}

.feedback-correct {{
    border-left-color: var(--active-color);
}}

.feedback-incorrect {{
    border-left-color: #F44336;
}}

.angle-display {{
    font-family: monospace;
    padding: 5px;
    border-radius: 3px;
    background-color: {("rgba(0,0,0,0.05)", "rgba(255,255,255,0.1)")[st.session_state.dark_mode]};
    margin: 2px 0;
}}
</style>
""",
    unsafe_allow_html=True,
)
