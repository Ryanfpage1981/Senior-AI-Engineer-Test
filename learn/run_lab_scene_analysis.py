import os
import argparse
import numpy as np
import cv2
from ultralytics import YOLO


def reset_active_tracks(active_tracks, current_frame, tracking_file):
    for track_id in list(active_tracks.keys()):
        # If a track is not in the current frame and hasn't been marked as lost yet
        time_alive = (
            active_tracks[track_id]["last_seen"] - active_tracks[track_id]["first_seen"]
        )
        tracking_file.write(
            f"################# TRACKING EVENT - LOST ITEM {track_id}  ##################\n"
        )
        tracking_file.write(
            f"Track {track_id} ({active_tracks[track_id]['class_name']}) lost at frame {current_frame}. Was alive for {time_alive} frames.\n"
        )
        del active_tracks[track_id]


def run_tacking_update(boxes, active_tracks, class_names, current_frame, tracking_file):
    track_ids = boxes.id.int().cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()

    current_tracks_info = {}
    for i, track_id in enumerate(track_ids):
        class_id = classes[i]
        class_name = class_names[class_id]
        confidence = confidences[i]

        current_tracks_info[track_id] = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
        }

    # Update active tracks
    for track_id, info in current_tracks_info.items():
        if track_id not in active_tracks:
            active_tracks[track_id] = {
                "first_seen": current_frame,
                "last_seen": current_frame,
                "class_id": info["class_id"],
                "class_name": info["class_name"],
                "confidences": [info["confidence"]],
            }
            ##New event this could be broadcast to any subscribers in a future design
            tracking_file.write(
                f"################# TRACKING EVENT - NEW ITEM {track_id} ##################\n"
            )
            tracking_file.write(f"New item: {active_tracks[track_id]}\n")
        else:
            active_tracks[track_id]["last_seen"] = current_frame
            active_tracks[track_id]["confidences"].append(info["confidence"])
            # If class changes (rare but possible), update it
            if active_tracks[track_id]["class_id"] != info["class_id"]:
                print(
                    f"Warning: Track {track_id} changed class from {active_tracks[track_id]['class_name']} to {info['class_name']} at frame {current_frame}"
                )
                active_tracks[track_id]["class_id"] = info["class_id"]
                active_tracks[track_id]["class_name"] = info["class_name"]

    # Check for lost tracks
    current_track_ids = list(current_tracks_info.keys())
    for track_id in list(active_tracks.keys()):
        # If a track is not in the current frame and hasn't been marked as lost yet
        if track_id not in current_track_ids:
            time_alive = (
                active_tracks[track_id]["last_seen"]
                - active_tracks[track_id]["first_seen"]
            )
            tracking_file.write(
                "################# TRACKING EVENT - LOST ITEM  ##################\n"
            )
            tracking_file.write(
                f"Track {track_id} ({active_tracks[track_id]['class_name']}) lost at frame {current_frame}. Was alive for {time_alive} frames.\n"
            )
            del active_tracks[track_id]


if __name__ == "__main__":
    print("Run scene analysis")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_file", default="./data/original/AICandidateTest-FINAL.mp4"
    )
    parser.add_argument(
        "--model_weights", default="./runs/detect/train/weights/best.pt"
    )
    parser.add_argument("--output_video_path", default="./results/tracking_overlay.mp4")
    parser.add_argument("--confidence_level", default=0.25)
    parser.add_argument("--display_in_real_time", default=False)
    args = parser.parse_args()
    results_dir = "./results"  ##add results dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # load model
    model = YOLO(args.model_weights)
    class_names = model.names
    # Open the video file
    cap = cv2.VideoCapture(args.video_file)

    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {input_video_file}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    ##open output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video_path, fourcc, fps, (width, height))

    tracking_evt_log = open(os.path.join(results_dir, "tracking_events.txt"), "w")
    # Dictionary to store active tracks
    active_tracks = {}
    frame_count = 0
    ##run over video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # results = model(frame, conf=args.confidence_level)
        results = model.track(
            frame,
            conf=args.confidence_level,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        # Get the current tracks
        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            run_tacking_update(
                boxes, active_tracks, class_names, frame_count, tracking_evt_log
            )
        else:
            if bool(active_tracks):
                reset_active_tracks(
                    active_tracks, frame_count, tracking_evt_log
                )  ##incase we have no tracks
        frame_count += 1

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        if args.display_in_real_time:
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    tracking_evt_log.close()
    cv2.destroyAllWindows()
