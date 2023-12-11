from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "traffic2.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

#https://github.com/ultralytics/ultralytics/issues/3399

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            # Get the boxes and track IDs
            else:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


#videos:
#https://atu-main-mdl-euwest1.s3.eu-west-1.amazonaws.com/cd/00/cd00f68bdb995ec2578fc97820ab283425200d7f?response-content-disposition=inline%3B%20filename%3D%22traffic.mp4%22&response-content-type=video%2Fmp4&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWRN6GJFLWCMOG6H7%2F20231211%2Feu-west-1%2Fs3%2Faws4_request&X-Amz-Date=20231211T153959Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21541&X-Amz-Signature=1b9f0cc0a235613b67f1c197a3d5525cae75c16728037e05f0f36ac153054dfd
#https://atu-main-mdl-euwest1.s3.eu-west-1.amazonaws.com/89/3d/893d1c71c17fd07a945b120eebe93ae926208772?response-content-disposition=inline%3B%20filename%3D%22traffic2.mp4%22&response-content-type=video%2Fmp4&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWRN6GJFLWCMOG6H7%2F20231211%2Feu-west-1%2Fs3%2Faws4_request&X-Amz-Date=20231211T154110Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21590&X-Amz-Signature=3c28198acdddd19a0656f73d0562f27bb48f64cd499f8f582105e4e32d830f3c
#https://atu-main-mdl-euwest1.s3.eu-west-1.amazonaws.com/0c/b7/0cb78a7d77f8276ea9aad03d3136828f26d9d639?response-content-disposition=inline%3B%20filename%3D%22traffic3.mp4%22&response-content-type=video%2Fmp4&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWRN6GJFLWCMOG6H7%2F20231211%2Feu-west-1%2Fs3%2Faws4_request&X-Amz-Date=20231211T153955Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21545&X-Amz-Signature=e3bb03887ec3b4bcb97f85f66dfdca5766b5ba3659a8c13e0fba51debc2f4df5