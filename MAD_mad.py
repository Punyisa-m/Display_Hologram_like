import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to count fingers
def count_fingers(hand_landmarks, hand_label):
    finger_tips = [8, 12, 16, 20]  # Tips of index, middle, ring, and pinky fingers
    thumb_tip = 4  # Tip of thumb
    count = 0

    # Check thumb
    if hand_label == "Left" and hand_landmarks[thumb_tip].x > hand_landmarks[thumb_tip - 2].x:
        count += 1
    elif hand_label == "Right" and hand_landmarks[thumb_tip].x < hand_landmarks[thumb_tip - 2].x:
        count += 1

    # Check other fingers
    for tip in finger_tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            count += 1

    return count

# Function to check the direction of the index finger
def get_index_finger_direction(hand_landmarks):
    index_finger_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_mcp = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    if index_finger_tip.x < index_finger_mcp.x:
        return "LEFT"
    elif index_finger_tip.x > index_finger_mcp.x:
        return "RIGHT"
    return "STRAIGHT"

# Load the video files
video_files = ['Bluewhale_video.mp4','jellyfish_video.mp4', 'earth_output.mp4', 'many_fish_video.mp4']
videos = [cv2.VideoCapture(file) for file in video_files]
video_names = ['1','2','3','4']

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

selected_video = None
last_direction = None
frame_moved = 0
total_frames = [int(video.get(cv2.CAP_PROP_FRAME_COUNT)) for video in videos]

fingers_time_count = [0, 0, 0, 0]  # To store the time count for 1 to 8 fingers
required_seconds = 1  # Time threshold in seconds to change video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.flip(frame, 1)
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    result = hands.process(frame_rgb)
    hand_status = {'Left': False, 'Right': False}
    total_fingers = 0
    hand_labels = {'Left': None, 'Right': None}

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            hand_labels[hand_label] = hand_landmarks.landmark
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            fingers_count = count_fingers(hand_landmarks.landmark, hand_label)
            total_fingers += fingers_count

            # Update hand status
            if fingers_count == 0:
                hand_status[hand_label] = True

        # Check for "STOP" condition
        if hand_status['Left'] and hand_status['Right']:
            action_text = 'STOP'
            last_direction = 'STOP'
        elif hand_status['Left']:
            action_text = 'Right'
            last_direction = 'LEFT'
        elif hand_status['Right']:
            action_text = 'LEFT'
            last_direction = 'RIGHT'
        else:
            action_text = ''
            last_direction = None

        # Display the number of fingers and action
        cv2.putText(frame, f'Fingers: {total_fingers}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if action_text:
            cv2.putText(frame, action_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if action_text == 'TURN LEFT' else (255, 0, 0) if action_text == 'TURN RIGHT' else (0, 0, 255), 2, cv2.LINE_AA)

        if total_fingers in [1, 2, 3, 4]:
            fingers_time_count[total_fingers - 1] += 1
            if fingers_time_count[total_fingers - 1] >= required_seconds * 2:  # Assuming 30 FPS
                selected_video = total_fingers - 1
                fingers_time_count = [0, 0, 0, 0]  # Reset the counts
        else:
            fingers_time_count = [0, 0, 0, 0]  # Reset if other than 1 to 8 fingers

        if  selected_video is not None:
            if last_direction == 'RIGHT':
                frame_moved = (frame_moved - 1) % total_frames[selected_video]
            elif last_direction == 'LEFT':
                frame_moved = (frame_moved + 1) % total_frames[selected_video]

            videos[selected_video].set(cv2.CAP_PROP_POS_FRAMES, frame_moved)
            ret_video, frame_video = videos[selected_video].read()

            if ret_video:
                cv2.namedWindow(' ', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(' ', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow(' ', frame_video)
                #cv2.imshow('Selected Video', frame_video)
            # else:
            #     if selected_video is not None:
            #         frame_moved = (frame_moved + 1) % total_frames[selected_video]
            #         videos[selected_video].set(cv2.CAP_PROP_POS_FRAMES, frame_moved)
            #         ret_video, frame_video = videos[selected_video].read()
            #         # if ret_video:
            #         #     cv2.imshow('', frame_video)

    # Create a combined frame to show all videos with titles
    combined_frame = None
    for i, video in enumerate(videos):
        ret, video_frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, video_frame = video.read()

        video_frame = cv2.resize(video_frame, (frame.shape[1] // 3, frame.shape[0] // 3))

        # Add title to video frame
        cv2.putText(video_frame, video_names[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if combined_frame is None:
            combined_frame = video_frame
        else:
            combined_frame = cv2.hconcat([combined_frame, video_frame])

    # Show the combined frame with all videos
    cv2.imshow('Video Catalog', combined_frame)

    # Show the frame with hand tracking
    cv2.imshow('Hand Tracking', frame)

    # Check for keypress to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
for video in videos:
    video.release()
cv2.destroyAllWindows()