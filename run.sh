#!/bin/bash


# ./run.sh input.mp4 calib.json results --gpu

# --- parse args -------------------------------------------------------------
VIDEO="$1"
CALIB="$2"
OUT_DIR="$3"      
GPU_FLAG=""
KALMAN_FLAG=""

for arg in "$@"; do
  case $arg in
    --gpu) GPU_FLAG="--gpu" ;;
    --kalman) KALMAN_FLAG="--kalman" ;;
  esac
done

# --- environment setup -------------------------------------------------------
echo "[run.sh] Setting up Python virtual environment..."

# # 1. Force the script to recognize pyenv
# export PATH="$HOME/.pyenv/bin:$PATH"
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)" # Only if you use pyenv-virtualenv

# 2. Tell pyenv to use 3.10.14 for this script
pyenv shell 3.10.14

if [ ! -d ".venv" ]; then
    echo "[run.sh] Creating new virtual environment with Python 3.10..."
    # USE 'python' HERE, NOT 'python3.10'
    python -m venv .venv
fi

# 3. Activate the environment
source .venv/bin/activate

echo "[run.sh] Installing Python dependencies..."
pip install -q -r requirements.txt


# --- Weights Management ------------------------------------------------------
# FIX: Explicitly define the weights directory name
WEIGHTS_DIR="weights"
WEIGHTS_FILE="$WEIGHTS_DIR/best.pt"  
DRIVE_FILE_ID="1ajsGmaSMl95B7mOwj_n4NFWzj2UShsa7"

# Create the folder first
mkdir -p "$WEIGHTS_DIR"

# Download if missing (removed --id to fix warning)
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "[run.sh] Model weights not found. Downloading from Google Drive..."
    gdown "$DRIVE_FILE_ID" -O "$WEIGHTS_FILE"
else
    echo "[run.sh] Model weights ($WEIGHTS_FILE) already exist."
fi

# --- CUDA Verification -------------------------------------------------------
if [[ -n "$GPU_FLAG" ]]; then
  echo "[run.sh] Verifying CUDA availability..."
  python -c "import torch; assert torch.cuda.is_available(), 'ERROR: --gpu flag used, but CUDA not found!'"
fi

# --- run pipeline -----------------------------------------------------------
echo "[run.sh] Starting tracker..."
echo "[run.sh] Video   : $VIDEO"
echo "[run.sh] Calib   : $CALIB"
echo "[run.sh] Out Dir : $OUT_DIR"
echo "[run.sh] Weights : $WEIGHTS_FILE"
[[ -n "$GPU_FLAG" ]] && echo "[run.sh] Mode    : GPU" || echo "[run.sh] Mode    : CPU"
[[ -n "$KALMAN_FLAG" ]] && echo "[run.sh] Kalman  : enabled" || echo "[run.sh] Kalman  : disabled"
echo ""

# Execute Python script 
# (Make sure track_bin.py accepts the --weights argument!)
python track_bin.py \
  --video "$VIDEO" \
  --calib "$CALIB" \
  --output "$OUT_DIR" \
  --weights "$WEIGHTS_FILE" \
  $GPU_FLAG \
  $KALMAN_FLAG
