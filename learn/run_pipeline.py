#!/usr/bin/env python
import subprocess
import sys
import os
import time


def run_script(script_path):
    """Run a Python script and exit if it fails."""
    print(f"\n{'='*80}\nRunning {script_path}\n{'='*80}\n")
    start_time = time.time()

    result = subprocess.run([sys.executable, script_path])

    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed:.2f} seconds with exit code {result.returncode}")

    if result.returncode != 0:
        print(f"Error running {script_path}. Exiting pipeline.")
        sys.exit(result.returncode)

    return True


if __name__ == "__main__":

    # List of scripts to run in order
    scripts = [
        "learn/run_image_extractor.py",
        "learn/run_yolo_img_gen.py",
        "learn/run_yolo_fine_tune.py",
        "learn/run_lab_scene_analysis.py",
    ]

    # Run each script in sequence
    for script in scripts:
        success = run_script(script)
        if not success:
            break

    print("\nPipeline completed successfully!")
