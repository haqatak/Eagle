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

if torch.backends.mps.is_available():
    device = torch.device("mps")

# --- (NumPy JSON Encoders - Unchanged) ---
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

# --- Annotation Function (Restored to full) ---
# Now that annotations are synced, we can show keypoints again
def annotate_frame(frame, frame_data, team_mapping, current_metadata):
    cv2.putText(frame, current_metadata, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5, cv2.LINE_AA)
    if not frame_data:
        return frame

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

    # We can try drawing all balls again, as they should be accurate
    if "Ball" in frame_data.get("Coordinates", {}):
        for ball_id, data in frame_data["Coordinates"]["Ball"].items():
            if "Bottom_center" not in data: continue
            x, y = data["Bottom_center"]
            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), -1) # Yellow

    # Re-enable keypoints - they should be correct now
    if "Keypoints" in frame_data:
        for point in frame_data.get("Keypoints", {}).values():
            if point is not None:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 0), -1)

    return frame

# --- (Aggregation Function - Unchanged) ---
def aggregate_frame_data(buffer):
    if not buffer:
        return {}
    # ... (rest of function is unchanged) ...
    sums = {"Coordinates": {}, "Keypoints": {}}
    counts = {"Coordinates": {}, "Keypoints": {}}
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

def format_aggregated_data(agg_data, timestamp_str):
    """
    Formats the raw aggregated data into the user's specified JSON structure.
    """

    # 1. Format Timestamp (e.g., "2025-11-02T18:54:15Z")
    timestamp = f"{timestamp_str}Z"

    # 2. Format Boundaries (from Keypoints)
    # This creates a list of [x,y] coords for the field corners
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
                # Round coordinates for a cleaner JSON
                coords = [round(c, 2) for c in data["Bottom_center"]]
                coordinates_list.append({
                    "ID": entity_id,
                    "Coordinates": coords,
                    "Type": entity_type
                })

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

# --- (Metadata Listener - Unchanged) ---
def metadata_listener(hls_url: str, metadata_queue: queue.Queue, stop_event: threading.Event):
    print("Metadata listener started. Waiting for stream...")
    # ... (rest of function is unchanged) ...
    last_sent_time_str = None
    stream_was_live = False

    while not stop_event.is_set():
        try:
            with av.open(hls_url, 'r', options={'rw_timeout': '5000000'}, timeout=5) as container:
                stream_was_live = True
                print("Metadata listener: Stream found!")
                data_streams = [s for s in container.streams if s.type == 'data']
                if not data_streams:
                    print("Metadata listener: No data streams found. Retrying...")
                    time.sleep(2)
                    continue

                pattern = re.compile(b'TXXX\x00\x00.*ID3-TIME:([^\x00]+)')
                for packet in container.demux(data_streams):
                    if stop_event.is_set():
                        break
                    if packet.size > 0:
                        raw_data = bytes(packet)
                        match = pattern.search(raw_data)
                        if match:
                            time_str = match.group(1).decode('utf-8', errors='ignore')
                            if time_str != last_sent_time_str:
                                metadata_queue.put(time_str)
                                last_sent_time_str = time_str

            if stream_was_live:
                print("Metadata listener: Stream ended. Stopping.")
                stop_event.set()
                break

        except av.error.FileNotFoundError:
            if stream_was_live:
                print("Metadata listener: Stream file gone. Stopping.")
                stop_event.set()
                break
            else:
                time.sleep(2)

        except Exception as e:
            if stop_event.is_set():
                break
            print(f"Metadata listener: Error: {e}. Retrying in 5s.")
            time.sleep(5)
            last_sent_time_str = None

    print("Metadata listener: Stopped.")


# --- NEW: Thread 1 - Reader Thread ---
def frame_reader(cap, job_queue, native_fps, stop_event):
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

        # --- MODIFIED LOGIC ---
        # This is an "overwrite" put. It ensures the queue
        # always contains the *latest* frame.
        try:
            job_queue.put_nowait(job_data) # Try to put the new frame in
        except queue.Full:
            try:
                job_queue.get_nowait() # If full, remove the old frame
            except queue.Empty:
                pass # The worker got it just in time
            job_queue.put_nowait(job_data) # Now put the new frame in
        # --- END MODIFICATION ---

        # --- Framerate Limiter ---
        elapsed_sec = time.time() - loop_start_time
        wait_sec = target_delay_sec - elapsed_sec
        if wait_sec > 0:
            time.sleep(wait_sec)

    print("Reader thread: Stopping.")
    job_queue.put(None) # Signal worker to stop

# --- NEW: Thread 2 - Worker Thread ---
def process_worker(model, processor, job_queue, result_queue, stop_event):
    """
    Worker processes a frame, annotates it, and sends the
    *annotated frame* and *raw data* to the main thread.
    """
    print("Worker thread started.")
    current_metadata_for_worker = "Processing..." # Local cache

    while not stop_event.is_set():
        try:
            # Get a job, but with a timeout so we can check the stop_event
            job_data = job_queue.get(timeout=1)

            if job_data is None: # Shutdown signal
                break

            frame, fps, frame_index = job_data

            # --- The slow processing ---
            frame_data = model.process_single_frame(frame.copy(), fps=fps)
            processor.update(frame.copy(), frame_data)
            team_mapping = processor.get_team_mapping()
            # --- Done ---

            # --- Get latest metadata from main (via the processor, just an idea)
            # This is complex. Let's just use a placeholder for now.
            # We will update the metadata in the main thread.

            # We send back:
            # 1. The original frame
            # 2. The raw data (for aggregation)
            # 3. The processed data (for display)
            processed_data_for_display = (frame, frame_data, team_mapping, frame_index)

            # Put the result in the display queue
            result_queue.put(processed_data_for_display, timeout=1)
            job_queue.task_done()

        except queue.Empty:
            # This is normal, means job_queue was empty
            continue
        except queue.Full:
            # This is bad, means main display thread is stuck
            print("Worker: Result queue full. Display thread is stuck?")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in worker thread: {e}")

    print("Worker thread: Stopping.")
    result_queue.put(None) # Signal main display to stop

# --- NEW: Thread 3 - Main/Display Thread ---
def main_realtime():
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, default="0", help="Path to the video file or 'HLS stream")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files.")
    args = parser.parse_args()

    # --- Queue Setup ---
    # We set maxsize to create "back-pressure" and define our buffer
    # e.g., maxsize=50 means ~2 seconds of buffer at 25 FPS
    job_queue = queue.Queue(maxsize=1)      # Reader -> Worker
    result_queue = queue.Queue(maxsize=50)   # Worker -> Main/Display
    metadata_queue = queue.Queue()          # Listener -> Main/Display
    stop_event = threading.Event()

    # --- Main Thread Setup (Video Capture) ---
    print(f"Main: Waiting for video stream at {args.video_path}...")
    cap = cv2.VideoCapture(args.video_path)

    while not cap.isOpened() and not stop_event.is_set():
        time.sleep(0.5)
        cap.release()
        cap = cv2.VideoCapture(args.video_path)

        cv2.imshow("Waiting for stream...", np.zeros((200, 400), dtype=np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Main: Quit during wait.")
            stop_event.set()
            break

    if not cap.isOpened():
        print("Main: Could not open stream. Exiting.")
        stop_event.set()
        return

    cv2.destroyWindow("Waiting for stream...")
    print("Main: Video stream found! Starting processing.")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps == 0 or native_fps > 1000:
        native_fps = 25.0
    print(f"Processing at {native_fps} FPS")

    # --- Models ---
    model = CoordinateModel()
    processor = RealTimeProcessor(fps=native_fps)

    # --- Start All Threads ---
    listener_thread = threading.Thread(target=metadata_listener,
                                       args=(args.video_path, metadata_queue, stop_event),
                                       daemon=True)
    reader_thread = threading.Thread(target=frame_reader,
                                     args=(cap, job_queue, native_fps, stop_event))
    worker_thread = threading.Thread(target=process_worker,
                                     args=(model, processor, job_queue, result_queue, stop_event))

    listener_thread.start()
    reader_thread.start()
    worker_thread.start()

    # --- Video/Data Saving Setup ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = None
    all_frames_for_video = []
    all_aggregated_data_dict = {}

    # --- State ---
    current_second_buffer = []
    current_metadata = "Waiting for metadata..."
    current_date = datetime.now().strftime("%Y-%m-%d")
    agg_frame_count = 0
    worker_has_produced_first_frame = False

    # --- MAIN DISPLAY LOOP ---
    start_time = time.time()
    while not stop_event.is_set():
        try:
            # --- 1. Check for HLS Metadata (Triggers Aggregation) ---
            try:
                new_metadata = metadata_queue.get_nowait()
                current_metadata = new_metadata
                print(f"Main: Got new metadata: {new_metadata}")

                if current_second_buffer and worker_has_produced_first_frame:
                    print(f"Main: Aggregating {len(current_second_buffer)} frames...")
                    aggregated_data_raw = aggregate_frame_data(current_second_buffer)

                    timestamp_str = f"{current_date}T{current_metadata}"
                    formatted_data = format_aggregated_data(aggregated_data_raw, timestamp_str)

                    frame_key = f"frame_{agg_frame_count:05d}"
                    all_aggregated_data_dict[frame_key] = formatted_data

                    agg_frame_count += 1
                    current_second_buffer.clear()
            except queue.Empty:
                pass # No new metadata

            # --- 2. Get Finished Frame from Worker ---
            # This is the main blocking call.
            # We wait here for the worker to give us a frame to show.
            processed_data = result_queue.get(timeout=1)

            if processed_data is None: # Shutdown signal
                print("Main: Received shutdown signal from worker.")
                stop_event.set()
                break

            original_frame, raw_frame_data, team_mapping, frame_index = processed_data

            if not worker_has_produced_first_frame:
                worker_has_produced_first_frame = True
                print("Main: Worker is ready. Aggregation enabled.")

            # Add raw data to aggregation buffer
            current_second_buffer.append(raw_frame_data)

            # --- 3. Annotate and Display ---
            status_text = f"Frame: {frame_index} | Time: {current_metadata}"
            annotated_frame = annotate_frame(original_frame,
                                             raw_frame_data,
                                             team_mapping,
                                             status_text)

            cv2.imshow("Eagle Real-Time Tracking", annotated_frame)
            all_frames_for_video.append(annotated_frame)

            result_queue.task_done()

            # --- 4. Check for Quit Key ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Main: 'q' key pressed. Shutting down.")
                stop_event.set()
                break

        except queue.Empty:
            # This is normal, just means no new frame is ready from worker
            # The display will just freeze for a moment
            if not reader_thread.is_alive() or not worker_thread.is_alive():
                print("Main: A processing thread died. Shutting down.")
                stop_event.set()
            continue

    # --- Cleanup ---
    print("Shutting down...")
    end_time = time.time()
    stop_event.set() # Signal all threads

    # Wait for threads to finish
    reader_thread.join(timeout=2)
    worker_thread.join(timeout=5) # Worker may need more time

    cap.release()

    if current_second_buffer:
        print(f"Main: Aggregating final {len(current_second_buffer)} frames...")

        # 1. Aggregate
        aggregated_data_raw = aggregate_frame_data(current_second_buffer)

        # 2. Format
        timestamp_str = f"{current_date}T{current_metadata}"
        formatted_data = format_aggregated_data(aggregated_data_raw, timestamp_str)

        # 3. Save
        frame_key = f"frame_{agg_frame_count:05d}"
        all_aggregated_data_dict[frame_key] = formatted_data
        agg_frame_count += 1

    # --- Save Demo Video ---
    if all_frames_for_video:

        # --- CALCULATE TRUE FPS ---
        total_duration_sec = end_time - start_time
        processed_frames_count = len(all_frames_for_video)
        processed_fps = native_fps # Default fallback

        if total_duration_sec > 0:
            processed_fps = processed_frames_count / total_duration_sec

        if processed_fps < 1: # Ensure FPS is at least 1
            processed_fps = 1
        # --- END CALCULATION ---

        if out_video is None:
            h, w, _ = all_frames_for_video[0].shape
            video_path = os.path.join(args.output_dir, "annotated_demo_video.mp4")
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            # --- USE THE TRUE FPS ---
            out_video = cv2.VideoWriter(video_path, fourcc, processed_fps, (w, h))
            print(f"Saving {processed_frames_count} frames to {video_path} at {processed_fps:.2f} FPS")

        for f in all_frames_for_video:
            out_video.write(f)

    if out_video:
        out_video.release()

    # --- Save Aggregated JSON ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    json_filename = os.path.join(args.output_dir, "aggregated_per_second_data.json")
    print(f"Saving {len(all_aggregated_data_dict)} aggregated data entries to {json_filename}")
    with open(json_filename, 'w') as f:
        dump(all_aggregated_data_dict, f, cls=NumpyEncoder, indent=4)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_realtime()