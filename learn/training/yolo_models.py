from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
"""
Training util functions for using YOLO models
"""


def test_train(path_to_conf, v8_type='n',img_w=1280):
    """
    Simple way to test everything is working with the current install 
    """
    model = YOLO(f'yolov8{v8_type}.pt') ##load pretrained weights
    try:
        results = model.train(data=path_to_conf, epochs=1, imgsz=img_w)
    except Exception as e:
        print("Error running test model")
        print(e)
        raise

def fine_tune_train(path_to_conf, layers_to_freeze, v8_type='n', epochs=50, img_w=1280):
    model = YOLO(f'yolov8{v8_type}.pt') ##load pretrained weights
    results = model.train(data=path_to_conf,
            epochs=epochs,
            imgsz=img_w, 
            freeze=layers_to_freeze)
    return results


