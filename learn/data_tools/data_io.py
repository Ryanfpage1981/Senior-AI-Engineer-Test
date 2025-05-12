import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np

"""
Provides a series of util funtions for reading and writing datasets. Includes a parser for CVATannotations. 
"""

def write_labels_to_file(img_labels, output_dir, file_prefix='image', convert_from_cvat_to_yolo=True, base_img_dim=[1280,720]):
    """
    Write labels using CVAT and optinally convert notation to YOLO before writing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(path_name, exist_ok=True)
    
    img_width, img_height = base_img_dim[:2]
    for i, img_l in enumerate(img_labels):

        file = os.path.join(output_dir, file_prefix + f"_{i}.txt") 
        with open(file, 'w') as f:
            for iroi in img_l:
                class_id = iroi[0]
                
                # Get coordinates (in CVAT format)
                xmin = iroi[1][0]
                ymin = iroi[1][1]
                xmax = iroi[1][2]
                ymax = iroi[1][3]
                if convert_from_cvat_to_yolo:
                    # Convert to YOLO format (normalized center coordinates and dimensions)
                    center_x = ((xmin + xmax) / 2) / img_width
                    center_y = ((ymin + ymax) / 2) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    # Write to YOLO format file
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                else:
                    f.write(f"{class_id} {xmin:.6f} {ymin:.6f} {xmax:.6f} {ymax:.6f}\n")


def extract_images_from_video(input_video_file, frame_ids, roi=None):

    frame_ids = sorted(frame_ids)
     
    # Open the video file
    cap = cv2.VideoCapture(input_video_file)

    if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {input_video_file}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
        
    if max(frame_ids) >= total_frames:
        raise ValueError(f"Error: Requested frame {max(frame_ids)} exceeds total frames {total_frames}")

    extracted_frames = []
    current_idx = 0
    frame_counter = 0
    while cap.isOpened() and current_idx < len(frame_ids):
        # Get the next frame to extract
        target_frame = frame_ids[current_idx]
        
        # If we're not yet at the target frame, use seek
        if frame_counter < target_frame:
            # Set position to target frame (faster than reading each frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            frame_counter = target_frame
        
        # Read the frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_counter}")
            break
        
        # Check if this is a frame we want to extract
        if frame_counter == target_frame:
            # Apply ROI if specified
            if roi is not None:
                xmin, ymin, xmax, ymax = roi[current_idx]
                frame = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
            
            extracted_frames.append(frame)
            
            
            # Move to the next frame in our list
            current_idx += 1
        
        # Increment frame counter
        frame_counter += 1

    cap.release()

    print(f"Extracted {len(extracted_frames)} frames out of {len(frame_ids)} requested")

    return extracted_frames

def write_images_to_file(images, output_file_path, file_name_prefix="image"):
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path, exist_ok=True)
    for i, img in enumerate(images):
        output_path = os.path.join(output_file_path, f"{file_name_prefix}_{i}.png")
        cv2.imwrite(output_path, img)
        print(f"Saved frame {output_path}")

def parse_cvat_annotations(input_file):
    """ 
    Find fist instance of unique id and return a list of class label,
    id and ROI. Except the hand here we are using all the samples annotated.
    """
    # Parse XML file
    tree = ET.parse(input_file)
    root = tree.getroot()
    class_data = {}    
    # Process each image
    for m in root.findall('.//meta'):
        image = m.find('original_size')
        img_width = float(image.find('width').text)
        img_height = float(image.find('height').text)
        print(f"Image width {img_width} {img_height}")
         
    # Process each bounding box
    for trk in root.findall('.//track'):
        class_label = str(trk.get('label'))
        class_id = int(trk.get('id'))
        print(f"Class {class_label}, id {class_id}") 
        if(class_label == "Hand"):
            for box in trk.findall('.//box'):
                # Get coordinates (in CVAT format)
                if int(box.get('occluded')) == 0:
                    xmin = float(box.get('xtl'))
                    ymin = float(box.get('ytl'))
                    xmax = float(box.get('xbr'))
                    ymax = float(box.get('ybr')) 
                    frame_id = int(box.get('frame'))
                    if class_label not in class_data:
                        class_data[class_label] = [[class_id, frame_id, [xmin, ymin, xmax, ymax]]]
                    else:
                        class_data[class_label].append([class_id, frame_id, [xmin, ymin, xmax, ymax]])

        else:
            box = trk.find('box')

            # Get coordinates (in CVAT format)
            xmin = float(box.get('xtl'))
            ymin = float(box.get('ytl'))
            xmax = float(box.get('xbr'))
            ymax = float(box.get('ybr')) 
            frame_id = int(box.get('frame'))
            if class_label not in class_data:
                class_data[class_label] = [[class_id, frame_id, [xmin, ymin, xmax, ymax]]]
            else:
                class_data[class_label].append([class_id, frame_id, [xmin, ymin, xmax, ymax]])

    return class_data


