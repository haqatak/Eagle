"""
Real-Time HLS Football Tracking and Aggregation Pipeline.

This script runs a multi-threaded application to process a live HLS video stream
containing timed ID3 metadata. It performs the following steps:

1.  Reader Thread: Reads video frames from the HLS stream at their native FPS.
2.  Metadata Thread: Listens for timed ID3 metadata (timestamps) from the stream.
3.  Worker Thread: Performs heavy model inference (object detection, keypoints)
    on frames provided by the reader.
4.  Main/Display Thread:
    a) Receives finished, processed frames from the worker.
    b) Displays the annotated video and a real-time minimap.
    c) Receives metadata "ticks" to trigger data aggregation.
    d) Aggregates 1 second of data and writes it incrementally to a JSON file.
"""

import base64
from datetime import datetime
import os
import re
import threading
import time
import cv2
import av
import numpy as np
from argparse import ArgumentParser
import json
import queue
import torch

from eagle.models.coordinate_model import CoordinateModel
from eagle.processor import RealTimeProcessor

# --- Global Configuration ---

# Set maxsize=1 to create a "last-frame" buffer (minimal latency)
# Set maxsize > 1 to create a "first-in, first-out" buffer (smoother video, more latency)
JOB_QUEUE_SIZE = 1

# Buffer for finished frames. A larger size means smoother playback but
# more memory usage and potential shutdown delay.
RESULT_QUEUE_SIZE = 50

# Fallback FPS if it cannot be read from the video stream
DEFAULT_FPS = 25.0

# Size of the "Waiting for stream..." popup window
WAITING_WINDOW_SIZE = (400, 200)

if torch.backends.mps.is_available():
    device = torch.device("mps")

# --- NumPy JSON Encoders ---

class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy data types. """
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
    """ Custom JSON decoder for NumPy data types. """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

# --- JSON dump/load wrappers ---

def dumps(*args, **kwargs):
    """ json.dumps wrapper using the NumpyEncoder. """
    kwargs.setdefault('cls', NumpyEncoder)
    return json.dumps(*args, **kwargs)

def loads(*args, **kwargs):
    """ json.loads wrapper using the Numpy decoder. """
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.loads(*args, **kwargs)

def dump(*args, **kwargs):
    """ json.dump wrapper using the NumpyEncoder. """
    kwargs.setdefault('cls', NumpyEncoder)
    # Fix: We do not call .copy() on the tuple 'args'
    return json.dump(*args, **kwargs)

def load(*args, **kwargs):
    """ json.load wrapper using the Numpy decoder. """
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.load(*args, **kwargs)

# --- Annotation and Drawing Functions ---

def annotate_frame(frame, frame_data, team_mapping, current_metadata):
    """
    Draws all annotations (players, ball, keypoints, metadata) on a video frame.

    Args:
        frame (np.ndarray): The video frame to draw on.
        frame_data (dict): The raw detection data from the model.
        team_mapping (dict): A dict mapping player_id to team_id (0 or 1).
        current_metadata (str): The metadata string to display.

    Returns:
        np.ndarray: The annotated frame.
    """
    cv2.putText(frame, current_metadata, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5, cv2.LINE_AA)
    if not frame_data:
        return frame

    # Draw Players and Goalkeepers
    for entity_type in ["Player", "Goalkeeper"]:
        if entity_type in frame_data.get("Coordinates", {}):
            for player_id, data in frame_data["Coordinates"][entity_type].items():
                if "Bottom_center" not in data: continue
                x, y = data["Bottom_center"]
                team_id = team_mapping.get(player_id)
                color = (0, 255, 0) # Default green
                if team_id == 0:
                    color = (255, 0, 0) # Blue
                elif team_id == 1:
                    color = (0, 0, 255) # Red
                cv2.ellipse(frame, (int(x), int(y)), (35, 18), 0, -45, 235, color, 2)
                cv2.putText(frame, str(player_id), (int(x) - 10, int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw Ball(s)
    if "Ball" in frame_data.get("Coordinates", {}):
        for ball_id, data in frame_data["Coordinates"]["Ball"].items():
            if "Bottom_center" not in data: continue
            x, y = data["Bottom_center"]
            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), -1) # Yellow

    # Draw Field Keypoints
    if "Keypoints" in frame_data:
        for point in frame_data.get("Keypoints", {}).values():
            if point is not None:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 0), -1)

    return frame

def draw_minimap(minimap_canvas, dst_points, frame_data, team_mapping):
    """
    Draws the top-down minimap on a canvas using a perspective transform.

    Args:
        minimap_canvas (np.ndarray): The map (a green-filled array) to draw on.
        dst_points (np.ndarray): The 4 destination points for the map.
        frame_data (dict): The raw detection data (must contain Keypoints).
        team_mapping (dict): A dict mapping player_id to team_id (0 or 1).

    Returns:
        np.ndarray: The annotated minimap.
    """
    # 1. Reset the map to a fresh green
    minimap_canvas[:] = (0, 100, 0) # Dark green

    # 2. Get the 4 corner points from the video frame
    kp = frame_data.get("Keypoints", {})
    src_points_list = [
        kp.get("Bottom_left"),
        kp.get("Top_left"),
        kp.get("Top_right"),
        kp.get("Bottom_right")
    ]

    # 3. If we don't have all 4 corners, we can't draw the map.
    if any(p is None for p in src_points_list):
        cv2.putText(minimap_canvas, "No Keypoints", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return minimap_canvas

    src_points = np.array(src_points_list, dtype=np.float32)

    # 4. Calculate the Perspective Transform Matrix
    try:
        M = cv2.getPerspectiveTransform(src_points, dst_points)
    except cv2.error:
        # Occurs if points are co-linear
        cv2.putText(minimap_canvas, "Bad Keypoints", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return minimap_canvas

    # 5. Collect all player and ball coordinates to be transformed
    coords_to_transform = []
    colors_to_draw = []

    # Get players
    for entity_type in ["Player", "Goalkeeper"]:
        for player_id, data in frame_data.get("Coordinates", {}).get(entity_type, {}).items():
            if "Bottom_center" in data:
                coords_to_transform.append([data["Bottom_center"]])
                team_id = team_mapping.get(player_id)
                if team_id == 0:
                    colors_to_draw.append((255, 0, 0)) # Blue
                elif team_id == 1:
                    colors_to_draw.append((0, 0, 255)) # Red
                else:
                    colors_to_draw.append((0, 255, 0)) # Green

    # Get ball(s)
    for ball_id, data in frame_data.get("Coordinates", {}).get("Ball", {}).items():
        if "Bottom_center" in data:
            coords_to_transform.append([data["Bottom_center"]])
            colors_to_draw.append((0, 255, 255)) # Yellow

    if not coords_to_transform:
        return minimap_canvas # No one to draw

    # 6. Apply the perspective transform
    np_coords = np.array(coords_to_transform, dtype=np.float32)
    transformed_coords = cv2.perspectiveTransform(np_coords, M)

    # 7. Draw the new coordinates on the minimap
    for i, (x_y_pair) in enumerate(transformed_coords):
        new_x, new_y = x_y_pair[0]
        color = colors_to_draw[i]
        cv2.circle(minimap_canvas, (int(new_x), int(new_y)), 7, color, -1)

    return minimap_canvas

# --- Data Processing Functions ---

def aggregate_frame_data(buffer):
    """
    Averages all coordinates from a buffer of frame_data dictionaries.
    This function is called once per second on all data processed in that second.

    Args:
        buffer (list): A list of raw `frame_data` dictionaries from the worker.

    Returns:
        dict: A single dictionary with the averaged data.
    """
    if not buffer:
        return {}

    sums = {"Coordinates": {}, "Keypoints": {}}
    counts = {"Coordinates": {}, "Keypoints": {}}

    # 1. Accumulate sums and counts for all entities
    for frame_data in buffer:
        for entity_type, entities in frame_data.get("Coordinates", {}).items():
            if entity_type not in sums["Coordinates"]:
                sums["Coordinates"][entity_type] = {}
                counts["Coordinates"][entity_type] = {}
            for entity_id, data in entities.items():
                if entity_id not in sums["Coordinates"][entity_type]:
                    sums["Coordinates"][entity_type][entity_id] = {
                        "Bottom_center": [0.0, 0.0],
                        "BBox": [0.0, 0.0, 0.0, 0.0]
                    }
                    counts["Coordinates"][entity_type][entity_id] = {
                        "Bottom_center": 0,
                        "BBox": 0
                    }
                pos = data.get("Bottom_center")
                if pos is not None:
                    sums["Coordinates"][entity_type][entity_id]["Bottom_center"][0] += pos[0]
                    sums["Coordinates"][entity_type][entity_id]["Bottom_center"][1] += pos[1]
                    counts["Coordinates"][entity_type][entity_id]["Bottom_center"] += 1
                bbox = data.get("BBox")
                if bbox is not None:
                    sums["Coordinates"][entity_type][entity_id]["BBox"][0] += bbox[0]
                    sums["Coordinates"][entity_type][entity_id]["BBox"][1] += bbox[1]
                    sums["Coordinates"][entity_type][entity_id]["BBox"][2] += bbox[2]
                    sums["Coordinates"][entity_type][entity_id]["BBox"][3] += bbox[3]
                    counts["Coordinates"][entity_type][entity_id]["BBox"] += 1

        for keypoint_name, pos in frame_data.get("Keypoints", {}).items():
            if keypoint_name not in sums["Keypoints"]:
                sums["Keypoints"][keypoint_name] = [0.0, 0.0]
                counts["Keypoints"][keypoint_name] = 0
            if pos is not None:
                sums["Keypoints"][keypoint_name][0] += pos[0]
                sums["Keypoints"][keypoint_name][1] += pos[1]
                counts["Keypoints"][keypoint_name] += 1

    # 2. Calculate final averages
    avg_data = {"Coordinates": {}, "Keypoints": {}}
    for entity_type, entities in sums["Coordinates"].items():
        avg_data["Coordinates"][entity_type] = {}
        for entity_id, data_sums in entities.items():
            avg_entity_data = {}
            count_bc = counts["Coordinates"][entity_type][entity_id]["Bottom_center"]
            if count_bc > 0:
                avg_entity_data["Bottom_center"] = (data_sums["Bottom_center"][0] / count_bc, data_sums["Bottom_center"][1] / count_bc)
            count_bbox = counts["Coordinates"][entity_type][entity_id]["BBox"]
            if count_bbox > 0:
                avg_entity_data["BBox"] = [int(data_sums["BBox"][0] / count_bbox), int(data_sums["BBox"][1] / count_bbox), int(data_sums["BBox"][2] / count_bbox), int(data_sums["BBox"][3] / count_bbox)]
            if avg_entity_data:
                avg_data["Coordinates"][entity_type][entity_id] = avg_entity_data
    for keypoint_name, pos_sum in sums["Keypoints"].items():
        count = counts["Keypoints"][keypoint_name]
        if count > 0:
            avg_data["Keypoints"][keypoint_name] = (pos_sum[0] / count, pos_sum[1] / count)

    return avg_data

def format_aggregated_data(agg_data, timestamp_str, team_mapping):
    """
    Formats the raw aggregated data into the user's specified JSON structure.

    Args:
        agg_data (dict): The averaged data from `aggregate_frame_data`.
        timestamp_str (str): The timestamp for this aggregation (e.g., "YYYY-MM-DDTHH:MM:SS:FF").
        team_mapping (dict): A dict mapping player_id to team_id (0 or 1).

    Returns:
        dict: A dictionary formatted to the user's specific JSON schema.
    """

    # 1. Format Timestamp (e.g., "2025-11-02T18:54:15Z")
    timestamp = f"{timestamp_str}Z"

    # 2. Format Boundaries (from Keypoints)
    boundaries = []
    kp = agg_data.get("Keypoints", {})
    boundaries.append(list(kp.get("Bottom_left", [0,0])))
    boundaries.append(list(kp.get("Top_left", [0,0])))
    boundaries.append(list(kp.get("Top_right", [0,0])))
    boundaries.append(list(kp.get("Bottom_right", [0,0])))

    # 3. Format Coordinates (into a flat list)
    coordinates_list = []

    # Process Players & Goalkeepers
    for entity_type in ["Player", "Goalkeeper"]:
        for entity_id, data in agg_data.get("Coordinates", {}).get(entity_type, {}).items():
            if "Bottom_center" in data:
                coords = [round(c, 2) for c in data["Bottom_center"]]
                item = {
                    "ID": entity_id,
                    "Coordinates": coords,
                    "Type": entity_type
                }
                if entity_type == "Player":
                    team_id = team_mapping.get(entity_id)
                    if team_id is not None:
                        item["Team"] = team_id
                coordinates_list.append(item)

    # Process Ball(s) - average all detected balls into a single "Ball" entry
    ball_sum = [0.0, 0.0]
    ball_count = 0
    for ball_id, data in agg_data.get("Coordinates", {}).get("Ball", {}).items():
        if "Bottom_center" in data:
            ball_sum[0] += data["Bottom_center"][0]
            ball_sum[1] += data["Bottom_center"][1]
            ball_count += 1

    if ball_count > 0:
        avg_ball_coords = [round(ball_sum[0] / ball_count, 2), round(ball_sum[1] / ball_count, 2)]
        coordinates_list.append({
            "ID": "Ball",
            "Coordinates": avg_ball_coords
        })

    # 4. Create Coordinates_video (as a copy, since we only have one coord system)
    coordinates_video_list = [
        {**item, "Coordinates": list(item["Coordinates"])}
        for item in coordinates_list
    ]

    # 5. Build the final dictionary
    output = {
        "Timestamp": timestamp,
        "Boundaries": boundaries,
        "Coordinates": coordinates_list,
        "Coordinates_video": coordinates_video_list
    }
    return output

# --- Thread Target Functions ---

def metadata_listener(hls_url: str, metadata_queue: queue.Queue, stop_event: threading.Event, is_file: bool = False):
    """
    Thread target: Listens for ID3 metadata tags in the HLS stream.

    - Retries connection if the stream is not found.
    - De-bounces timestamps to only send *new* timestamps.
    - Signals the main `stop_event` if the stream ends.
    """
    print("Metadata listener started.")

    if is_file:
        print("Metadata listener: File mode. Generating synthetic metadata.")
        start_time = datetime.now()
        last_second = -1
        while not stop_event.is_set():
             # In file mode, we just generate a timestamp every second
             elapsed = (datetime.now() - start_time).total_seconds()
             current_second = int(elapsed)
             if current_second > last_second:
                 time_str = datetime.now().strftime("%H:%M:%S:00")
                 metadata_queue.put(time_str)
                 last_second = current_second
             time.sleep(0.1)
        print("Metadata listener: Stopped.")
        return

    print("Metadata listener: Waiting for stream...")
    last_sent_time_str = None
    stream_was_live = False

    while not stop_event.is_set():
        try:
            # Open the stream with a 5-second timeout
            with av.open(hls_url, 'r', options={'rw_timeout': '5000000'}, timeout=5) as container:
                stream_was_live = True
                print("Metadata listener: Stream found!")
                data_streams = [s for s in container.streams if s.type == 'data']
                if not data_streams:
                    print("Metadata listener: No data streams found. Retrying...")
                    time.sleep(2)
                    continue

                pattern = re.compile(b'TXXX\x00\x00.*ID3-TIME:([^\x00]+)')

                # Demux the stream and look for ID3 packets
                for packet in container.demux(data_streams):
                    if stop_event.is_set():
                        break
                    if packet.size > 0:
                        raw_data = bytes(packet)
                        match = pattern.search(raw_data)
                        if match:
                            time_str = match.group(1).decode('utf-8', errors='ignore')
                            # De-bouncing: Only put new, unique timestamps in the queue
                            if time_str != last_sent_time_str:
                                metadata_queue.put(time_str)
                                last_sent_time_str = time_str

            # If the 'with' block exits, the stream ended
            if stream_was_live:
                print("Metadata listener: Stream ended. Stopping.")
                stop_event.set()
                break

        except av.error.FileNotFoundError:
            if stream_was_live:
                # If we were live and the file disappears, the stream is over.
                print("Metadata listener: Stream file gone. Stopping.")
                stop_event.set()
                break
            else:
                # Stream hasn't started yet, just wait.
                time.sleep(2)

        except Exception as e:
            if stop_event.is_set():
                break
            print(f"Metadata listener: Error: {e}. Retrying in 5s.")
            time.sleep(5)
            last_sent_time_str = None

    print("Metadata listener: Stopped.")


def frame_reader(cap, job_queue, native_fps, stop_event):
    """
    Thread target: Reads frames from the video capture at the native FPS.

    - Implements a framerate limiter to read at 25 FPS.
    - Uses an "overwrite" put logic to ensure the job_queue (of size 1)
      always contains the most recent frame, minimizing latency.
    """
    print("Reader thread started.")
    target_delay_sec = 1.0 / native_fps
    i = 0
    while not stop_event.is_set():
        loop_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Reader: cap.read() returned False. Stopping.")
            stop_event.set()
            break

        i += 1
        job_data = (frame, native_fps, i)

        # Overwrite logic: Ensures the worker always gets the *latest* frame
        try:
            job_queue.put_nowait(job_data) # Put new frame in
        except queue.Full:
            try:
                job_queue.get_nowait() # Remove old frame
            except queue.Empty:
                pass # Worker grabbed the old frame just in time
            job_queue.put_nowait(job_data) # Put new frame in

        # Framerate Limiter
        elapsed_sec = time.time() - loop_start_time
        wait_sec = target_delay_sec - elapsed_sec
        if wait_sec > 0:
            time.sleep(wait_sec)

    print("Reader thread: Stopping.")
    job_queue.put(None) # Signal worker to stop

def process_worker(model, processor, job_queue, result_queue, stop_event):
    """
    Thread target: The main "bottleneck" thread.

    - Gets a frame from the job_queue.
    - Runs the slow model inference (`process_single_frame`).
    - Runs the processor to get team mappings.
    - Puts the (original frame + all processed data) into the result_queue.
    """
    print("Worker thread started.")

    while not stop_event.is_set():
        try:
            # Wait for a job, but with a timeout to check stop_event
            job_data = job_queue.get(timeout=1)

            if job_data is None: # Shutdown signal
                break

            frame, fps, frame_index = job_data

            # --- The slow processing ---
            frame_data = model.process_single_frame(frame.copy(), fps=fps)
            processor.update(frame.copy(), frame_data)
            team_mapping = processor.get_team_mapping()
            # --- Done ---

            # Send all data needed by the main thread
            processed_data_for_display = (frame, frame_data, team_mapping, frame_index)

            # Put the result in the display queue
            result_queue.put(processed_data_for_display, timeout=1)
            job_queue.task_done()

        except queue.Empty:
            # This is normal, means job_queue was empty. Loop and check stop_event.
            continue
        except queue.Full:
            # This is bad, means main display thread is stuck
            print("Worker: Result queue full. Display thread is stuck?")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in worker thread: {e}")

    print("Worker thread: Stopping.")
    result_queue.put(None) # Signal main display to stop

# --- Main Application ---

def main_realtime():
    """
    Main function to set up and run the processing pipeline.

    - Parses arguments
    - Sets up queues and threads
    - Waits for the video stream
    - Starts all threads
    - Runs the main display and aggregation loop
    - Cleans up and saves all outputs
    """
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, default="0", help="Path to the video file or 'HLS stream")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    args = parser.parse_args()

    # --- 1. Setup Queues and Stop Event ---
    job_queue = queue.Queue(maxsize=JOB_QUEUE_SIZE)
    result_queue = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
    metadata_queue = queue.Queue()
    stop_event = threading.Event()

    # --- 2. Wait for Video Stream ---
    print(f"Main: Waiting for video stream at {args.video_path}...")
    cap = cv2.VideoCapture(args.video_path)

    # Check if input is a local file
    is_file = os.path.isfile(args.video_path)

    while not cap.isOpened() and not stop_event.is_set():
        if is_file:
             # If it's a file and we can't open it, it's an error.
             print(f"Main: Could not open file {args.video_path}")
             stop_event.set()
             return

        time.sleep(0.5)
        cap.release()
        cap = cv2.VideoCapture(args.video_path)

        if not args.headless:
            # Show a dummy window to allow quitting with 'q'
            wait_img = np.zeros(WAITING_WINDOW_SIZE, dtype=np.uint8)
            cv2.putText(wait_img, "Waiting for stream...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Waiting for stream...", wait_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Main: Quit during wait.")
                stop_event.set()
                break

    if not cap.isOpened():
        print("Main: Could not open stream. Exiting.")
        stop_event.set()
        return

    if not args.headless:
        cv2.destroyWindow("Waiting for stream...")
    print("Main: Video stream found! Starting processing.")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps == 0 or native_fps > 1000:
        native_fps = DEFAULT_FPS
    print(f"Processing at {native_fps} FPS")

    # --- 3. Initialize Models and Minimap ---
    model = CoordinateModel()
    processor = RealTimeProcessor(fps=native_fps)

    # Minimap setup
    MAP_HEIGHT = 600
    MAP_WIDTH = 400
    minimap_canvas = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
    dst_points = np.array([
        [0, MAP_HEIGHT], [0, 0], [MAP_WIDTH, 0], [MAP_WIDTH, MAP_HEIGHT]
    ], dtype=np.float32)

    # --- 4. Start All Threads ---
    listener_thread = threading.Thread(target=metadata_listener,
                                       args=(args.video_path, metadata_queue, stop_event, is_file),
                                       daemon=True)
    reader_thread = threading.Thread(target=frame_reader,
                                     args=(cap, job_queue, native_fps, stop_event))
    worker_thread = threading.Thread(target=process_worker,
                                     args=(model, processor, job_queue, result_queue, stop_event))

    listener_thread.start()
    reader_thread.start()
    worker_thread.start()

    # --- 5. Setup Output Files ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = None
    all_frames_for_video = []
    all_aggregated_data_dict = {}

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    json_filename = os.path.join(args.output_dir, "aggregated_per_second_data.json")
    print(f"Saving aggregated JSON incrementally to {json_filename}")

    # --- 6. State for Main Loop ---
    current_second_buffer = []
    current_metadata = "Waiting for metadata..."
    current_date = datetime.now().strftime("%Y-%m-%d") # Bug fix: Initialize current_date
    agg_frame_count = 0
    worker_has_produced_first_frame = False

    # --- 7. MAIN DISPLAY LOOP ---
    start_time = time.time()
    while not stop_event.is_set():
        try:
            # --- Check for HLS Metadata (Triggers Aggregation) ---
            try:
                new_metadata = metadata_queue.get_nowait()
                current_metadata = new_metadata
                print(f"Main: Got new metadata: {new_metadata}")

                if current_second_buffer and worker_has_produced_first_frame:
                    print(f"Main: Aggregating {len(current_second_buffer)} frames...")
                    aggregated_data_raw = aggregate_frame_data(current_second_buffer)

                    # Bug fix: Add current_date to the timestamp
                    timestamp_str = f"{current_date}T{current_metadata}"
                    formatted_data = format_aggregated_data(aggregated_data_raw, timestamp_str, team_mapping)

                    frame_key = f"frame_{agg_frame_count:05d}"
                    all_aggregated_data_dict[frame_key] = formatted_data

                    # Save to file incrementally
                    with open(json_filename, 'w') as f:
                        dump(all_aggregated_data_dict, f, cls=NumpyEncoder, indent=4)

                    agg_frame_count += 1
                    current_second_buffer.clear()
            except queue.Empty:
                pass # No new metadata

            # --- Get Finished Frame from Worker ---
            # This is the main blocking call.
            processed_data = result_queue.get(timeout=1)

            if processed_data is None: # Shutdown signal
                print("Main: Received shutdown signal from worker.")
                stop_event.set()
                break

            original_frame, raw_frame_data, team_mapping, frame_index = processed_data

            # Set the "warmup" flag on the first frame
            if not worker_has_produced_first_frame:
                worker_has_produced_first_frame = True
                print("Main: Worker is ready. Aggregation enabled.")

            # Add raw data to aggregation buffer
            current_second_buffer.append(raw_frame_data)

            # --- Annotate and Display ---
            status_text = f"Frame: {frame_index} | Time: {current_metadata}"

            if args.headless:
                # Print progress to stdout so it can be monitored by the web interface
                print(f"Progress: {status_text}")

            annotated_frame = annotate_frame(original_frame,
                                             raw_frame_data,
                                             team_mapping,
                                             status_text)

            minimap_canvas = draw_minimap(minimap_canvas, dst_points, raw_frame_data, team_mapping)

            if not args.headless:
                cv2.imshow("Minimap", minimap_canvas)
                cv2.imshow("Eagle Real-Time Tracking", annotated_frame)

            all_frames_for_video.append(annotated_frame)

            result_queue.task_done()

            # --- Check for Quit Key ---
            if not args.headless:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Main: 'q' key pressed. Shutting down.")
                    stop_event.set()
                    break

        except queue.Empty:
            # No new frame is ready from the worker.
            # Check if threads are still alive.
            if not reader_thread.is_alive() or not worker_thread.is_alive():
                print("Main: A processing thread died. Shutting down.")
                stop_event.set()
            continue # Continue loop to check stop_event again

    # --- 8. Cleanup ---
    print("Shutting down...")
    end_time = time.time()
    stop_event.set() # Signal all threads

    # Wait for threads to finish
    reader_thread.join(timeout=2)
    worker_thread.join(timeout=5)

    cap.release()

    # --- Final aggregation flush ---
    if current_second_buffer:
        print(f"Main: Aggregating final {len(current_second_buffer)} frames...")
        aggregated_data_raw = aggregate_frame_data(current_second_buffer)

        # Bug fix: Add current_date to the timestamp
        timestamp_str = f"{current_date}T{current_metadata}"
        formatted_data = format_aggregated_data(aggregated_data_raw, timestamp_str, team_mapping)

        frame_key = f"frame_{agg_frame_count:05d}"
        all_aggregated_data_dict[frame_key] = formatted_data
        agg_frame_count += 1

        # Write the final version of the file
        print(f"Saving final aggregated entry to {json_filename}")
        with open(json_filename, 'w') as f:
            dump(all_aggregated_data_dict, f, cls=NumpyEncoder, indent=4)

    # --- Save Demo Video (with correct FPS) ---
    if all_frames_for_video:
        total_duration_sec = end_time - start_time
        processed_frames_count = len(all_frames_for_video)
        processed_fps = DEFAULT_FPS # Fallback

        if total_duration_sec > 0:
            processed_fps = processed_frames_count / total_duration_sec
        if processed_fps < 1:
            processed_fps = 1

        if out_video is None:
            h, w, _ = all_frames_for_video[0].shape
            video_path = os.path.join(args.output_dir, "annotated_demo_video.mp4")
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            out_video = cv2.VideoWriter(video_path, fourcc, processed_fps, (w, h))
            print(f"Saving {processed_frames_count} frames to {video_path} at {processed_fps:.2f} FPS")

        for f in all_frames_for_video:
            out_video.write(f)

    if out_video:
        out_video.release()

    if not args.headless:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_realtime()