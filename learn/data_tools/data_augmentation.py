import cv2
import numpy as np

"""
Augmentation for image and labels. 

Currently only doing a simple overlay.
"""


def compositor(bkg_img, fgnd_img, centre):
    """
    Simple scene compositor that adds a foreground image to the
    background.
    """
    # Get dimensions of the foreground image
    fgnd_height, fgnd_width = fgnd_img.shape[:2]

    # Calculate the position to place the foreground image
    # Centre coordinates represent where the center of the foreground should be placed
    x_pos_min = int(centre[0] - fgnd_width // 2)
    x_pos_max = int(centre[0] + fgnd_width // 2)
    y_pos_min = int(centre[1] - fgnd_height // 2)
    y_pos_max = int(centre[1] + fgnd_height // 2)

    # Account for odd-sized dimensions
    if fgnd_width % 2 == 1:
        x_pos_max += 1
    if fgnd_height % 2 == 1:
        y_pos_max += 1
    bkg_height, bkg_width = bkg_img.shape[:2]

    # Create a copy of the background image to avoid modifying the original
    result_img = bkg_img.copy()

    try:
        ##TODO Handle overlaying better so out of bounds do not throw exceptions
        result_img[y_pos_min:y_pos_max, x_pos_min:x_pos_max] = fgnd_img
        bounding_box = [x_pos_min, y_pos_min, x_pos_max, y_pos_max]
    except Exception as e:
        raise

    return result_img, bounding_box
