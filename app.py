import os
import shutil
import subprocess
import threading
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uuid
import time
from pathlib import Path

app = FastAPI()

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Store process status
# key: task_id, value: status dict
tasks = {}

def monitor_process(task_id: str, process: subprocess.Popen, output_path: str):
    """
    Monitors a running process.
    """
    try:
        stdout, stderr = process.communicate()
        return_code = process.returncode

        # Log output for debugging
        with open(os.path.join(output_path, "stdout.log"), "w") as f:
            f.write(stdout if stdout else "")
        with open(os.path.join(output_path, "stderr.log"), "w") as f:
            f.write(stderr if stderr else "")

        if return_code == 0:
            tasks[task_id] = {"status": "completed", "output_dir": output_path}
        else:
            tasks[task_id] = {
                "status": "failed",
                "error": stderr,
                "stdout": stdout
            }
    except Exception as e:
        tasks[task_id] = {"status": "error", "message": str(e)}

def start_processing(task_id: str, video_path: str):
    output_path = os.path.join(OUTPUT_DIR, task_id)
    os.makedirs(output_path, exist_ok=True)

    tasks[task_id] = {"status": "running", "message": "Processing started"}

    try:
        cmd = [
            "python", "main.py",
            "--video_path", video_path,
            "--output_dir", output_path,
            "--headless"
        ]

        print(f"Running command: {' '.join(cmd)}")

        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Start a thread to monitor it (wait for it)
        thread = threading.Thread(target=monitor_process, args=(task_id, process, output_path))
        thread.start()

    except Exception as e:
        tasks[task_id] = {"status": "error", "message": str(e)}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eagle Football Tracking</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { border: 1px solid #ccc; padding: 20px; border-radius: 5px; }
            .status { margin-top: 20px; padding: 10px; background-color: #f0f0f0; }
            .error { color: red; }
            .success { color: green; }
        </style>
    </head>
    <body>
        <h1>Eagle Football Tracking</h1>
        <div class="container">
            <h2>Upload Video</h2>
            <form action="/process" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="video/*" required>
                <button type="submit">Process Video</button>
            </form>

            <h2>Or Input Stream URL</h2>
            <form action="/process_url" method="post">
                <input type="text" name="url" placeholder="http://example.com/stream.m3u8" required style="width: 300px;">
                <button type="submit">Process Stream</button>
            </form>
        </div>

        <div id="status-container"></div>

        <script>
            // Simple polling script could go here
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    file_location = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")

    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    start_processing(task_id, file_location)

    return {"task_id": task_id, "message": "Processing started", "status_url": f"/status/{task_id}"}

@app.post("/process_url")
async def process_url(url: str):
    task_id = str(uuid.uuid4())
    start_processing(task_id, url)
    return {"task_id": task_id, "message": "Processing started", "status_url": f"/status/{task_id}"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.get("/results/{task_id}/json")
async def get_json_result(task_id: str):
    if task_id not in tasks:
         raise HTTPException(status_code=404, detail="Task not found")

    if tasks[task_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail=f"Result not ready. Status: {tasks[task_id]['status']}")

    path = os.path.join(tasks[task_id]["output_dir"], "aggregated_per_second_data.json")
    if os.path.exists(path):
        return FileResponse(path, media_type='application/json', filename="data.json")
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/results/{task_id}/video")
async def get_video_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    if tasks[task_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail=f"Result not ready. Status: {tasks[task_id]['status']}")

    path = os.path.join(tasks[task_id]["output_dir"], "annotated_demo_video.mp4")
    if os.path.exists(path):
        return FileResponse(path, media_type='video/mp4', filename="video.mp4")
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/logs/{task_id}")
async def get_logs(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    output_path = os.path.join(OUTPUT_DIR, task_id)
    logs = {}
    if os.path.exists(os.path.join(output_path, "stdout.log")):
        with open(os.path.join(output_path, "stdout.log"), "r") as f:
            logs["stdout"] = f.read()
    if os.path.exists(os.path.join(output_path, "stderr.log")):
        with open(os.path.join(output_path, "stderr.log"), "r") as f:
            logs["stderr"] = f.read()

    return logs
