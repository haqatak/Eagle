import base64
from datetime import datetime
import os
import re
import threading

import cv2
import av
import numpy as np
from argparse import ArgumentParser
import json
import queue
import torch

from eagle.models.coordinate_model import CoordinateModel
from eagle.processor import RealTimeProcessor

if torch.backends.mps.is_available():
    device = torch.device("mps")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(obj.tobytes()).decode('utf-8')
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)

        if isinstance(obj, np.generic):
            return obj.item()

        return super().default(obj)

def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def dumps(*args, **kwargs):
    kwargs.setdefault('cls', NumpyEncoder)
    return json.dumps(*args, **kwargs)

def loads(*args, **kwargs):
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.loads(*args, **kwargs)

def dump(*args, **kwargs):
    kwargs.setdefault('cls', NumpyEncoder)
    return json.dump(*args, **kwargs)

def load(*args, **kwargs):
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.load(*args, **kwargs)

def annotate_frame(frame, frame_data, team_mapping, current_metadata):
    cv2.putText(frame, current_metadata, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5, cv2.LINE_AA)

    if not frame_data:
        return frame

    for entity_type in ["Player", "Goalkeeper"]:
        if entity_type in frame_data["Coordinates"]:
            for player_id, data in frame_data["Coordinates"][entity_type].items():
                x, y = data["Bottom_center"]
                team_id = team_mapping.get(player_id)

                if team_id == 0:
                    color = (255, 0, 0)
                elif team_id == 1:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.ellipse(frame, (int(x), int(y)), (35, 18), 0, -45, 235, color, 2)
                cv2.putText(frame, str(player_id), (int(x) - 10, int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if "Ball" in frame_data["Coordinates"]:
        for ball_id, data in frame_data["Coordinates"]["Ball"].items():
            x, y = data["Bottom_center"]
            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), -1) # Yellow circle for the ball

    if "Keypoints" in frame_data:
        for point in frame_data["Keypoints"].values():
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 0), -1)

    return frame

def metadata_listener(hls_url: str, metadata_queue: queue.Queue):
    try:
        with av.open(hls_url, 'r', timeout=10) as container:
            data_streams = [s for s in container.streams if s.type == 'data']
            if not data_streams:
                print("No data streams found in the HLS source.")
                return

            pattern = re.compile(b'TXXX\x00\x00.*ID3-TIME:([^\x00]+)')

            for packet in container.demux(data_streams):
                if packet.size > 0:
                    raw_data = bytes(packet)
                    match = pattern.search(raw_data)

                    if match:
                        time_str = match.group(1).decode('utf-8', errors='ignore')
                        print(time_str)
                        # stream_timestamp = packet.pts * packet.time_base
                        # parsed_data = {'type': 'time_update', 'value': time_str}
                        metadata_queue.put(time_str)

    except Exception as e:
        print(f"Error in metadata listener: {e}")

def main_realtime():
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, default="0", help="Path to the video file or 'HLS stream")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files.")
    args = parser.parse_args()

    metadata_queue = queue.Queue()
    listener_thread = threading.Thread(target=metadata_listener, args=(args.video_path, metadata_queue), daemon=True)
    listener_thread.start()

    model = CoordinateModel()
    cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video source at {args.video_path}")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps == 0:
        native_fps = 25  # Default for webcams

    processor = RealTimeProcessor(fps=native_fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = None
    all_frames_data = {}
    current_metadata = "No active metadata"
    all_frames = []

    # current date is not encoded in given id3 metadata stream todo
    current_date = datetime.now().strftime("%Y-%m-%d")

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        frame_data = model.process_single_frame(frame, fps=native_fps)
        if frame_data:
            if i % native_fps == 0:
                try:
                    current_metadata = metadata_queue.get_nowait()
                except queue.Empty:
                    pass

            i += 1

            frame_data["TimeStamp"] = f"{current_date}T{current_metadata}"
            processor.update(frame, frame_data)
            team_mapping = processor.get_team_mapping()

            frame_key = f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):05d}"
            print(frame_key)
            all_frames_data[frame_key] = frame_data

            annotated_frame = annotate_frame(frame.copy(), frame_data, team_mapping, current_metadata)

            all_frames.append(annotated_frame)
            cv2.imshow("Eagle Real-Time Tracking with Metadata", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out_video is None:
        h, w, _ = all_frames[0].shape
        out_video = cv2.VideoWriter(os.path.join(args.output_dir, "annotated_video.mp4"), fourcc, native_fps, (w, h))

    for frame in all_frames:
        out_video.write(frame)

    if out_video:
        out_video.release()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    json_filename = os.path.join(args.output_dir, "all_frames_data.json")
    with open(json_filename, 'w') as f:
        dump(all_frames_data, f, cls=NumpyEncoder, indent=4)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_realtime()