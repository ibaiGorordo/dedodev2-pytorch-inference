from datetime import timedelta
import cv2
from cap_from_youtube import cap_from_youtube
from dedodev2.feature_tracker import FeatureTracker

# Initialize the FeatureTracker
tracker = FeatureTracker()

videoUrl = "https://youtu.be/LHW-19iFk1I?si=ULxAcsTrsitxyBxK"
cap = cap_from_youtube(videoUrl, start=timedelta(seconds=10))

while True:

    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break

    # Update the tracker with the current frame
    success, matches = tracker.update(frame)

    # Draw the tracked keypoints and their trajectories
    tracked_frame = tracker.draw_tracks(frame)

    # Display the frame
    cv2.imshow("Feature Tracker", tracked_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()