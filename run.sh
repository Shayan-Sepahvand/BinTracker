#!/bin/bash


# ./run.sh input.mp4 calib.json results weights --gpu

# --- parse args -------------------------------------------------------------
VIDEO="$1"
CALIB="$2"
OUT_DIR="$3"      
# Note: $4 is ignored now because we download weights automatically

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

if [ ! -d ".venv" ]; then
    echo "[run.sh] Creating new virtual environment with Python 3.10..."
    python3.10 -m venv .venv
fi

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





# #!/usr/bin/env bash
# # =============================================================================
# # Skyscouter – CV Engineer Assessment
# # Entry point. Run as: bash run.sh --video input.mp4 --calib calib.json
# # Add --gpu flag to enable GPU inference (document GPU specs in README).
# # =============================================================================
# set -e
# ``
# # --- parse arguments ---------------------------------------------------------
# VIDEO=""
# CALIB=""
# GPU_FLAG=""
# KALMAN_FLAG=""

# while [[ $# -gt 0 ]]; do
#   case $1 in
#     --video)   VIDEO="$2";       shift 2 ;;
#     --calib)   CALIB="$2";       shift 2 ;;
#     --gpu)     GPU_FLAG="--gpu"; shift   ;;
#     --kalman)  KALMAN_FLAG="--kalman"; shift ;;
#     *)         echo "Unknown argument: $1"; exit 1 ;;
#   esac
# done

# if [[ -z "$VIDEO" || -z "$CALIB" ]]; then
#   echo "Usage: bash run.sh --video <path> --calib <path> [--gpu] [--kalman]"
#   exit 1
# fi



# # --- run pipeline ------------------------------------------------------------
# echo "[run.sh] Starting tracker..."
# echo "[run.sh] Video : $VIDEO"
# echo "[run.sh] Calib : $CALIB"
# [[ -n "$GPU_FLAG"    ]] && echo "[run.sh] Mode  : GPU"    || echo "[run.sh] Mode  : CPU"
# [[ -n "$KALMAN_FLAG" ]] && echo "[run.sh] Kalman: enabled" || echo "[run.sh] Kalman: disabled"
# echo ""

# python track_bin.py \
#   --video  "$VIDEO"  \
#   --calib  "$CALIB"  \
#   $GPU_FLAG          \
#   $KALMAN_FLAG
