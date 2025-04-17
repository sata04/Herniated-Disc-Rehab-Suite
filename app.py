# Consolidate imports at top
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

# Configure log levels to suppress verbose output
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)  # MediaPipeのログを抑制
av.logging.set_level(av.logging.PANIC)

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

# Initialize posture correctness state
if "posture_ok" not in st.session_state:
    st.session_state.posture_ok = False

# ダークモード設定をローカルストレージから読み込むためのJavaScript
st.markdown(
    """
    <script>
    // Streamlitイベントリスナー
    window.addEventListener('DOMContentLoaded', (event) => {
        setTimeout(() => {
            // ローカルストレージからダークモード設定を取得
            const storedDarkMode = localStorage.getItem('dark_mode');
            if (storedDarkMode === 'true') {
                // ダークモードがtrueの場合、チェックボックスをクリックして同期
                const darkModeCheckbox = document.querySelector('input[data-testid="stCheckbox"]');
                if (darkModeCheckbox && !darkModeCheckbox.checked) {
                    darkModeCheckbox.click();
                }
            }
        }, 500); // ページ読み込み後少し遅延させる
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# Sidebar UI with improved styling
st.sidebar.title("Herniated Disc Rehab Suite")
st.sidebar.markdown("---")

# ダークモードをローカルストレージに保存/読み込みするJavaScript
st.markdown(
    """
    <script>
    // ページ読み込み時に前回設定を取得
    const dm = localStorage.getItem('dark_mode');
    if(dm !== null) document.getElementById('dark_mode').checked = (dm === 'true');
    </script>
    """,
    unsafe_allow_html=True,
)
# Sidebar App Settings
with st.sidebar.expander("⚙️ App Settings", expanded=False):
    # ダークモード設定が変更されたかを追跡するために古い値を保存
    old_dark_mode = st.session_state.dark_mode
    st.checkbox("Dark Mode", value=st.session_state.dark_mode, key="dark_mode", help="Toggle dark/light mode")

    # ダークモード設定が変更された場合、ローカルストレージに保存
    if old_dark_mode != st.session_state.dark_mode:
        st.markdown(
            f"""
            <script>
            // ダークモード設定をローカルストレージに保存
            localStorage.setItem('dark_mode', '{str(st.session_state.dark_mode).toLowerCase()}');
            </script>
            """,
            unsafe_allow_html=True,
        )

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
current_date = datetime.now().strftime("%Y年%m月%d日")
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
                self.exercise = StretchExercise(str(config_path))
                current_key = st.session_state.get("current_key", None)
                if current_key is not None:
                    self.exercise.set_stretch(current_key)

                # ...existing code...
                self.pose = mp.solutions.pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=model_complexity,  # ユーザー設定を反映
                    static_image_mode=False,  # 動画ストリーム向けトラッキング
                )
                # ...existing code...
                # State for phase and FPS calculation
                self.phase = st.session_state.get("exercise_phase", "idle")
                self.last_frame_time = time.time()
                # 評価状態を追加
                self.angle_status = {}
                self.is_pose_correct = False
                # 文字化け修正: UTF-8で正しく表示されるようにフィードバックテキストを変更
                self.feedback_text = "Detecting pose..."
                self.feedback_color = (255, 255, 255)
                self.hold_timer = 0
                self.hold_required = 2.0  # 正しい姿勢を保持すべき秒数
                self.last_eval_time = time.time()
                # 音声フィードバック状態を追加
                self.last_pose_state = False
                self.pose_state_changed = False
                self.audio_cooldown = 0
                self.last_audio_time = time.time()
                # 猶予期間の追跡を追加
                self.grace_start = None
                self.grace_period = 2.0  # 秒

            def _handle_grace_period(self, current_time):
                if self.grace_start is None:
                    self.grace_start = current_time
                elif current_time - self.grace_start > self.grace_period:
                    self.feedback_text = "Posture incorrect! Please adjust."
                    self.feedback_color = (0, 0, 255)  # 赤色
                    self.grace_start = None  # Reset grace period

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                # MediaPipe処理用にRGB変換＆writeableフラグ設定
                # mediapipeのパフォーマンス向上のための処理
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False

                # MediaPipeにROIディメンションを明示的に提供（警告解消のため）
                h, w = img.shape[:2]
                results = self.pose.process(img_rgb)

                img_rgb.flags.writeable = True
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # Update phase and calculate FPS
                self.phase = st.session_state.get("exercise_phase", "idle")
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_frame_time)
                self.last_frame_time = current_time

                # 'results' already obtained above

                # ポーズが検出される場合の処理
                if results.pose_landmarks:
                    # ランドマークを二重に描画してより目立たせる
                    # まず太く暗い線で描画
                    mp.solutions.drawing_utils.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(20, 180, 0), thickness=5, circle_radius=7),
                        mp.solutions.drawing_utils.DrawingSpec(color=(30, 30, 30), thickness=3, circle_radius=3),
                    )
                    # 次に細く明るい線で描画（輪郭効果）
                    mp.solutions.drawing_utils.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 250, 0), thickness=2, circle_radius=4),
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                    )

                    # 関節角度の評価とフレーム処理
                    # MediaPipeの画像処理結果からangle_statusを取得
                    processed_img, self.angle_status = self.exercise.process_frame(img.copy(), self.pose, results.pose_landmarks, return_angles=True)
                    # 角度評価結果を更新 - 条件が空の場合も考慮
                    if len(self.angle_status) > 0:
                        self.is_pose_correct = all(status.get("is_correct", False) for status in self.angle_status.values() if "is_correct" in status)
                    else:
                        self.is_pose_correct = False
                    # 処理されたイメージを使用
                    if len(self.angle_status) > 0:  # 角度情報が存在する場合のみ
                        img = processed_img

                    # ランドマークをより目立たせるため、各関節ポイントを強調表示
                    h, w = img.shape[:2]
                    for landmark in results.pose_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)  # 黄色の円で関節を強調

                    # タイマー開始前の準備状態の場合
                    if self.phase == "active" and not self.exercise.timer_started:
                        # 全ての角度条件が満たされているか確認
                        # self.is_pose_correct はすでに上で設定済み

                        # 姿勢状態が変わったことを検出
                        if self.is_pose_correct != self.last_pose_state:
                            self.pose_state_changed = True
                            self.last_pose_state = self.is_pose_correct
                        else:
                            self.pose_state_changed = False

                        if self.is_pose_correct:
                            # 猶予期間をリセット
                            self.grace_start = None

                            # 正しい姿勢を保持した時間を計測
                            if self.hold_timer == 0:
                                self.hold_timer = current_time
                                self.feedback_text = f"Hold position... {self.hold_required:.1f}s"
                                self.feedback_color = (0, 255, 255)  # 黄色
                            else:
                                hold_duration = current_time - self.hold_timer
                                # 必要な時間だけ姿勢を保持したらタイマー開始
                                if hold_duration >= self.hold_required:
                                    # タイマー開始
                                    success = self.exercise.start_timer()
                                    self.hold_timer = 0
                                    self.feedback_text = "Exercise started! Timer running"
                                    self.feedback_color = (0, 255, 0)  # 緑色

                                    # タイマー開始時の音声 - start.wavを再生
                                    if success:
                                        st.session_state.play_timer_start_sound = True
                                        # Sync UI timer start time when internal timer starts
                                        st.session_state.exercise_start_time = time.time()
                                else:
                                    # 保持中のカウントダウン表示
                                    remaining = self.hold_required - hold_duration
                                    self.feedback_text = f"Hold position... {remaining:.1f}s"
                                    self.feedback_color = (0, 255, 255)  # 黄色
                        else:
                            # 正しくない姿勢の場合は猶予期間を処理
                            self._handle_grace_period(current_time)

                            # 姿勢が正しくない場合はタイマーリセット
                            self.hold_timer = 0
                            self.feedback_text = "Get into the correct position"
                            self.feedback_color = (0, 165, 255)  # オレンジ

                    # タイマー動作中またはレスト中
                    elif self.phase == "active" and self.exercise.timer_started:
                        if self.exercise.is_resting:
                            self.feedback_text = "Rest period... Prepare for next set"
                            self.feedback_color = (0, 165, 255)  # オレンジ
                        else:
                            self.feedback_text = "Exercise in progress... Maintain posture"
                            self.feedback_color = (0, 255, 0)  # 緑色

                            # タイマー動作中に姿勢が崩れた場合も猶予期間を提供
                            if not self.is_pose_correct:
                                self._handle_grace_period(current_time)
                            else:
                                # 正しい姿勢に戻った場合は猶予期間をリセット
                                self.grace_start = None

                    # 姿勢評価とフィードバックの表示
                    cv2.putText(img, self.feedback_text, (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.feedback_color, 2)

                    # 各関節の角度表示（デバッグ情報）を追加
                    if st.session_state.get("show_angles", False):
                        y_pos = 100
                        for joint, data in self.angle_status.items():
                            if "angle" in data:
                                status_color = (0, 255, 0) if data.get("is_correct", False) else (0, 0, 255)
                                angle_text = f"{joint}: {data['angle']:.1f}° {'✓' if data.get('is_correct', False) else '✗'}"
                                cv2.putText(img, angle_text, (w - 250, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                                y_pos += 25

                    # エクササイズ/レスト表示
                    phase_label = "REST" if self.exercise.is_resting else "EXERCISE"
                    phase_color = (0, 165, 255) if self.exercise.is_resting else (0, 255, 0)
                    cv2.putText(img, phase_label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, phase_color, 4)

                    # FPS表示
                    cv2.putText(img, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    # 姿勢ステータス（OK/NG）表示
                    posture_text = "OK" if self.is_pose_correct else "NG"
                    posture_color = (0, 255, 0) if self.is_pose_correct else (0, 0, 255)
                    cv2.putText(img, f"Posture: {posture_text}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, posture_color, 3)

                    # 全セット完了チェックを追加
                    if self.exercise.current_rep > self.exercise.current_stretch.timer.repetitions:
                        # 全セット完了時のフィードバック表示
                        self.feedback_text = "All sets completed! Great work!"
                        self.feedback_color = (0, 255, 0)  # 緑色

                        # 音声フィードバック
                        if not hasattr(self, "all_complete_announced") or not self.all_complete_announced:
                            st.session_state.play_all_complete_sound = True
                            self.all_complete_announced = True
                else:
                    # ポーズが検出されない場合のプロンプト
                    h, w = img.shape[:2]
                    cv2.putText(img, "No pose detected", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(img, "Please stand in frame", (w // 2 - 170, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Streamlitのセッション状態と同期
                try:
                    st.session_state.current_rep = self.exercise.current_rep
                    st.session_state.exercise_phase = "rest" if self.exercise.is_resting else "active"
                    # Update overall posture correctness status for UI display
                    st.session_state.posture_ok = self.is_pose_correct
                except Exception as e:
                    print(f"Error updating session state: {e}")
                # Update overall posture correctness status for UI display
                st.session_state.posture_ok = self.is_pose_correct

                return av.VideoFrame.from_ndarray(img, format="bgr24")

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

# サウンドフィードバック機能の強化とリアルタイム通知
if st.session_state.current_key is not None:
    stretch_cfg = exercise.stretches[st.session_state.current_key]
    total_reps = stretch_cfg.timer.repetitions if stretch_cfg.timer.enabled else 0

    # セッション状態を初期化
    if "prev_rep" not in st.session_state:
        st.session_state.prev_rep = 0
    if "play_timer_start_sound" not in st.session_state:
        st.session_state.play_timer_start_sound = False

    # レップカウント変更時の音声
    prev = st.session_state.prev_rep
    curr = st.session_state.current_rep
    if curr != prev:
        if curr == 1 and prev == 0:
            # セッション開始音
            st.audio("sounds/start.wav")
        elif curr > prev and curr <= total_reps:
            # 1セット完了音
            st.audio("sounds/end.wav")
        elif prev > curr or (curr == total_reps and prev < total_reps):
            # 全セット終了音
            st.audio("sounds/allend.wav")

    # タイマー開始時の音声 (start.wavをタイマー開始音として使用)
    if st.session_state.play_timer_start_sound:
        st.audio("sounds/start.wav")
        st.session_state.play_timer_start_sound = False

    # 全セット完了時の音声フィードバック
    if st.session_state.get("play_all_complete_sound", False):
        st.audio("sounds/allend.wav")
        st.session_state.play_all_complete_sound = False

    st.session_state.prev_rep = curr

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
