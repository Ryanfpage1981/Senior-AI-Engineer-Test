import os
import argparse
import numpy as np
from training import yolo_models


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset_config", default='./data/yolo/lab_scene_dataset.yaml')
    args = parser.parse_args()
    conf = args.path_to_dataset_config
    print('Check Data Set and Yolo Package Install')
    try:
        yolo_models.test_train(conf)
    except Exception as e:
        print("Exit app, cant run test case")
        os.sys.exit(-1)

    print('Fine tune last layer on lab data set')
    freeze_layers = np.linspace(0, 21, 22, dtype=int).tolist()
    res = yolo_models.fine_tune_train(conf, freeze_layers)
