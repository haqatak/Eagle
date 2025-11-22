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
import viz

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

# Restore history on startup
def load_history():
    if not os.path.exists(OUTPUT_DIR):
        return

    for task_id in os.listdir(OUTPUT_DIR):
        task_dir = os.path.join(OUTPUT_DIR, task_id)
        if os.path.isdir(task_dir):
            # Check if successful
            if os.path.exists(os.path.join(task_dir, "annotated_demo_video.mp4")):
                tasks[task_id] = {"status": "completed", "output_dir": task_dir}
            elif os.path.exists(os.path.join(task_dir, "aggregated_per_second_data.json")):
                 # Partially done or failed
                 tasks[task_id] = {"status": "failed", "output_dir": task_dir, "message": "Interrupted or failed"}
            else:
                 # Likely empty or just started
                 pass

load_history()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Generate history HTML
    history_html = ""
    completed_tasks = {k: v for k, v in tasks.items() if v["status"] == "completed"}
    if completed_tasks:
        history_html += "<h2>Previous Runs</h2><ul>"
        for task_id, task in completed_tasks.items():
            history_html += f'<li><a href="/result/{task_id}">{task_id}</a> (Completed)</li>'
        history_html += "</ul>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eagle Football Tracking</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .container {{ border: 1px solid #ccc; padding: 20px; border-radius: 5px; }}
            .status {{ margin-top: 20px; padding: 10px; background-color: #f0f0f0; }}
            .error {{ color: red; }}
            .success {{ color: green; }}
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

        {history_html}

        <div id="status-container" class="status" style="display:none;">
            <p>Status: <span id="status-text">Waiting...</span></p>
            <p id="progress-container" style="display:none;">Progress: <span id="progress-text"></span></p>
            <p><a id="live-logs-link" href="#" target="_blank" style="display:none;">View Live Logs</a></p>
            <div id="results-links" style="display:none;">
                <p class="success">Processing Complete!</p>
                <p><a id="result-page-link" href="#" target="_blank" style="font-weight:bold; font-size:1.2em;">View Full Results Page</a></p>
            </div>
        </div>

        <script>
            const form = document.querySelector('form[action="/process"]');
            const urlForm = document.querySelector('form[action="/process_url"]');
            const statusContainer = document.getElementById('status-container');
            const statusText = document.getElementById('status-text');
            const progressContainer = document.getElementById('progress-container');
            const progressText = document.getElementById('progress-text');
            const resultsLinks = document.getElementById('results-links');
            const liveLogsLink = document.getElementById('live-logs-link');

            async function handleSubmit(e, action) {{
                e.preventDefault();
                const formData = new FormData(e.target);

                let url;
                let response;

                statusContainer.style.display = 'block';

                if (action === '/process_url') {{
                    const inputUrl = formData.get('url');
                    statusText.innerText = "Starting...";
                    url = `/process_url?url=${{encodeURIComponent(inputUrl)}}`;
                    response = await fetch(url, {{ method: 'POST' }});
                }} else {{
                    statusText.innerText = "Uploading...";
                    url = action;
                    response = await fetch(url, {{ method: 'POST', body: formData }});
                }}

                const data = await response.json();
                liveLogsLink.href = `/logs/${{data.task_id}}`;
                liveLogsLink.style.display = "inline";
                pollStatus(data.status_url, data.task_id);
            }}

            async function pollStatus(url, taskId) {{
                const interval = setInterval(async () => {{
                    try {{
                        const response = await fetch(url);
                        const data = await response.json();
                        statusText.innerText = data.status;

                        if (data.progress) {{
                            progressContainer.style.display = 'block';
                            progressText.innerText = data.progress;
                        }}

                        if (data.status === 'completed') {{
                            clearInterval(interval);
                            resultsLinks.style.display = 'block';
                            document.getElementById('result-page-link').href = `/result/${{taskId}}`;
                        }} else if (data.status === 'failed' || data.status === 'error') {{
                            clearInterval(interval);
                            statusText.innerText = "Failed: " + (data.message || data.error);
                        }}
                    }} catch (e) {{
                        console.error(e);
                    }}
                }}, 1000);
            }}

            form.addEventListener('submit', (e) => handleSubmit(e, '/process'));
            urlForm.addEventListener('submit', (e) => handleSubmit(e, '/process_url'));
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

@app.get("/results/{task_id}/voronoi")
async def get_voronoi(task_id: str):
    if task_id not in tasks or tasks[task_id]["status"] != "completed":
         raise HTTPException(status_code=404, detail="Result not ready or failed")

    output_dir = tasks[task_id]["output_dir"]
    json_path = os.path.join(output_dir, "aggregated_per_second_data.json")
    img_path = os.path.join(output_dir, "voronoi.png")

    if not os.path.exists(img_path):
        if not viz.generate_voronoi(json_path, img_path):
             raise HTTPException(status_code=500, detail="Failed to generate visualization")

    return FileResponse(img_path, media_type='image/png')

@app.get("/result/{task_id}", response_class=HTMLResponse)
async def result_page(task_id: str):
    if task_id not in tasks:
        return HTMLResponse("<h1>Task not found</h1>", status_code=404)

    task = tasks[task_id]
    if task["status"] != "completed":
        return HTMLResponse(f"<h1>Task is {task['status']}</h1><p>Please wait...</p>")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eagle Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
            .viz-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .viz-item {{ border: 1px solid #ccc; padding: 10px; text-align: center; }}
            img {{ max-width: 400px; height: auto; }}
            video {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>Processing Results</h1>
        <p><a href="/">Back to Home</a></p>

        <h2>Annotated Video</h2>
        <video controls>
            <source src="/results/{task_id}/video" type="video/mp4">
            Your browser does not support the video tag.
        </video>

        <h2>Visualizations</h2>
        <div class="viz-container">
            <div class="viz-item">
                <h3>Voronoi Diagram</h3>
                <img src="/results/{task_id}/voronoi" alt="Voronoi Diagram">
            </div>
            <div class="viz-item">
                <h3>Pass Plot</h3>
                <img src="/results/{task_id}/pass" alt="Pass Plot">
            </div>
            <div class="viz-item">
                <h3>Player Trajectory</h3>
                <img src="/results/{task_id}/trajectory" alt="Player Trajectory">
            </div>
        </div>

        <h2>Data</h2>
        <p><a href="/results/{task_id}/json" target="_blank">Download JSON Data</a></p>
    </body>
    </html>
    """
    return html

@app.get("/results/{task_id}/pass")
async def get_pass(task_id: str):
    if task_id not in tasks or tasks[task_id]["status"] != "completed":
         raise HTTPException(status_code=404, detail="Result not ready or failed")

    output_dir = tasks[task_id]["output_dir"]
    json_path = os.path.join(output_dir, "aggregated_per_second_data.json")
    img_path = os.path.join(output_dir, "pass.png")

    if not os.path.exists(img_path):
        if not viz.generate_pass_plot(json_path, img_path):
             raise HTTPException(status_code=500, detail="Failed to generate visualization")

    return FileResponse(img_path, media_type='image/png')

@app.get("/results/{task_id}/trajectory")
async def get_trajectory(task_id: str):
    if task_id not in tasks or tasks[task_id]["status"] != "completed":
         raise HTTPException(status_code=404, detail="Result not ready or failed")

    output_dir = tasks[task_id]["output_dir"]
    json_path = os.path.join(output_dir, "aggregated_per_second_data.json")
    img_path = os.path.join(output_dir, "trajectory.png")

    if not os.path.exists(img_path):
        if not viz.generate_trajectory(json_path, img_path):
             raise HTTPException(status_code=500, detail="Failed to generate visualization")

    return FileResponse(img_path, media_type='image/png')
