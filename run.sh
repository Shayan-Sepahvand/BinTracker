#!/bin/bash


# ./run.sh --video input.mp4 --calib calib.json --kalman

# --- parse args -------------------------------------------------------------
VIDEO=""
CALIB=""
OUT_DIR="results" 
GPU_FLAG=""
KALMAN_FLAG=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --video) 
      VIDEO="$2"   
      shift 2      
      ;;
    --calib) 
      CALIB="$2"   
      shift 2      
      ;;
    --out)
      OUT_DIR="$2" 
      shift 2
      ;;
    --gpu) 
      GPU_FLAG="--gpu" 
      shift 1        
      ;;
    --kalman) 
      KALMAN_FLAG="--kalman"
      shift 1
      ;;
    *) 
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

# --- environment setup -------------------------------------------------------
echo "[run.sh] Setting up Python virtual environment (pyenv and python 3.10 must have been installed)..."
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell 3.10
if [ ! -d ".venv" ]; then
    echo "[run.sh] Creating new virtual environment with Python 3.10..."
    python -m venv .venv
fi
source .venv/bin/activate
echo "[run.sh] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# --- Weights Management ------------------------------------------------------
WEIGHTS_DIR="weights"
WEIGHTS_FILE="$WEIGHTS_DIR/best.pt"  
DRIVE_FILE_ID="1ajsGmaSMl95B7mOwj_n4NFWzj2UShsa7"
mkdir -p "$WEIGHTS_DIR"
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


python track_bin.py \
  --video "$VIDEO" \
  --calib "$CALIB" \
  --output "$OUT_DIR" \
  --weights "$WEIGHTS_FILE" \
  $GPU_FLAG \
  $KALMAN_FLAG
