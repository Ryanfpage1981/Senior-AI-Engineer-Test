#!/bin/bash

# Build the Docker image
docker build -t lab-scene-understanding .

# Check if a script was provided
if [ "$#" -eq 0 ]; then
  echo "Usage: ./run-docker.sh <all, extract, dataset_gen, train, analysis, analysis_pretrained>"
  echo "   or: ./run-docker.sh shell    (to get an interactive shell)"
  exit 1
fi

if [ "$1" = "all" ]; then
  docker run -it --rm \
    --gpus all \
    --shm-size=8G \
    -v "$PWD":/Senior-AI-Engineer-Test \
    --user "$(id -u):$(id -g)" \
    lab-scene-understanding python learn/run_pipeline.py
  exit 0
fi

if [ "$1" = "extract" ]; then
  docker run -it --rm \
    --gpus all \
    --shm-size=8G \
    -v "$PWD":/Senior-AI-Engineer-Test \
    --user "$(id -u):$(id -g)" \
    lab-scene-understanding python learn/run_image_extractor.py
  exit 0
fi

if [ "$1" = "dataset_gen" ]; then
  docker run -it --rm \
    --gpus all \
    --shm-size=8G \
    -v "$PWD":/Senior-AI-Engineer-Test \
    --user "$(id -u):$(id -g)" \
    lab-scene-understanding python learn/run_yolo_img_gen.py
  exit 0
fi

if [ "$1" = "train" ]; then
  docker run -it --rm \
    --gpus all \
    --shm-size=8G \
    -v "$PWD":/Senior-AI-Engineer-Test \
    --user "$(id -u):$(id -g)" \
    lab-scene-understanding python learn/run_yolo_fine_tune.py
  exit 0
fi

if [ "$1" = "analysis" ]; then
  docker run -it --rm \
    --gpus all \
    --shm-size=8G \
    -v "$PWD":/Senior-AI-Engineer-Test \
    --user "$(id -u):$(id -g)" \
    lab-scene-understanding python learn/run_lab_scene_analysis.py
  exit 0
fi


if [ "$1" = "analysis_pretrained" ]; then
  docker run -it --rm \
    --gpus all \
    --shm-size=8G \
    -v "$PWD":/Senior-AI-Engineer-Test \
    --user "$(id -u):$(id -g)" \
    lab-scene-understanding python learn/run_lab_scene_analysis.py --model_weights runs/detect/pretrained/weights/best.pt
  exit 0
fi

# Check for the "shell" argument
if [ "$1" = "shell" ]; then
  # Run an interactive bash shell in the container with GPU support
  docker run -it --rm \
    --gpus all \
    --shm-size=8G \
    -v "$PWD":/Senior-AI-Engineer-Test \
    --user "$(id -u):$(id -g)" \
    lab-scene-understanding /bin/bash
  exit 0
fi
