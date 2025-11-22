#!/bin/bash

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux could not be found. Please install it."
    exit 1
fi

# Activate Virtual Environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        echo "Activating .venv..."
        source .venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        echo "Activating venv..."
        source venv/bin/activate
    else
        echo "No virtual environment found."
        if command -v uv &> /dev/null; then
            echo "Creating .venv with uv..."
            uv venv
            source .venv/bin/activate
        else
            echo "Creating .venv with python..."
            python3 -m venv .venv
            source .venv/bin/activate
        fi
    fi
else
    echo "Already in virtual environment: $VIRTUAL_ENV"
fi

# Install dependencies
echo "Installing dependencies..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

# Ensure weights are present
echo "Checking weights..."
mkdir -p eagle/models/weights
if [ ! -f "eagle/models/weights/detector_large_hd.pt" ] || [ ! -f "eagle/models/weights/keypoints_main.pth" ]; then
    echo "Downloading weights..."
    # Assuming gdown is installed via requirements or pip
    if ! command -v gdown &> /dev/null; then
         pip install gdown
    fi

    # Download using the script
    # Note: the original script puts files in the root 'weights' folder
    bash eagle/models/get_weights.sh

    # Move them to expected location
    cp weights/detector_large_hd.pt eagle/models/weights/
    cp weights/keypoints_main.pth eagle/models/weights/
fi

# Ensure ReID weights are available (BoxMOT needs it)
if [ ! -f "osnet_x0_25_msmt17.pt" ]; then
    echo "Checking for ReID weights..."
    # BoxMOT downloads this automatically usually, but if it failed earlier, we can try to ensure it exists.
    # For now, we assume BoxMOT handles it or the user provides it if network fails.
    # But since we might need it, we can copy it if it was downloaded to a cache.
    # No action for now, relying on library.
    true
fi


SESSION_NAME="eagle"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create a new session
tmux new-session -d -s $SESSION_NAME -n "Web Server"

# Start the web server in the first window
# Removed --reload to prevent restarts when output files change
# We must also activate the environment inside the tmux session
if [ -f ".venv/bin/activate" ]; then
    tmux send-keys -t $SESSION_NAME:0 "source .venv/bin/activate" C-m
elif [ -f "venv/bin/activate" ]; then
    tmux send-keys -t $SESSION_NAME:0 "source venv/bin/activate" C-m
fi
tmux send-keys -t $SESSION_NAME:0 "uvicorn app:app --host 0.0.0.0 --port 8000" C-m

# Create a new window for potential ffmpeg streaming or other tasks
tmux new-window -t $SESSION_NAME:1 -n "Terminal"
if [ -f ".venv/bin/activate" ]; then
    tmux send-keys -t $SESSION_NAME:1 "source .venv/bin/activate" C-m
elif [ -f "venv/bin/activate" ]; then
    tmux send-keys -t $SESSION_NAME:1 "source venv/bin/activate" C-m
fi
tmux send-keys -t $SESSION_NAME:1 "echo 'Use this terminal for manual tasks'" C-m

echo "Started tmux session '$SESSION_NAME'."
echo "Web interface running at http://localhost:8000"
echo "Attach to session with: tmux attach -t $SESSION_NAME"
