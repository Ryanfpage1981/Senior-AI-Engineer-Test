# Run Lab Scene Analysis

To run the full pipeline first copy the video into the dir data/original with file name AICandidateTest-FINAL.mp4. Then run:

```./run_lab_scene_understanding.sh all```

This will build the image then run through all stages of the pipeline, populating the results dir with a video overlay and tracking event log. It will also create a dataset that sits in /data and when it runs the fine tune it will add training plots runs/detect/train. This is also were the model weights that were used are stored.

This should the GPU for training and inference. My host machine setup is Ubuntu 22.04.5, docker version 28.1.1 and NVIDIA-SMI 535.247.01 Driver Version: 535.247.01 CUDA Version: 12.2, hopefully the docker setup enables GPU utilising on the machine you are running on.

If that does not work and you want to run just the analysis you can use

```./run_lab_scene_understanding.sh analysis_pretrained```

This will use the model weights I committed into the repo. If you want to look into each component in more detail then you can either jump in with shell and run each python script with custom args or run each component seperatly. It does not take additional args into the script to allow each component to be configure currently. One of the many things I need to add. 

Please reach out if you want more details on any piece. 
