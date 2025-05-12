import argparse
import os
from data_tools import data_io
"""
App to run image extraction on video using CVAT annotation to define ROIs.
Also extracts background image, using frame 0 from the video.
"""


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvat_file", default="data/orignal/AICandidateTest-FINAL.mp4")
    parser.add_argument("--video_file", default="data/cvat_annotations/labels_run_0_2000.xml")
    parser.add_argument("--bkg_image_path", default="data/orignal/bkg_images")
    parser.add_argument("--class_data_image_path", default="data/orignal/")
    parser.add_argument("--run_bkg_extraction", action="store_true")
    args = parser.parse_args()
    input_video_file=args.video_file
    if args.run_bkg_extraction:
        output_bkg_image_path=args.bkg_image_path 
        
        print("Extract background image for data augmentation")
        bkg_image = data_io.extract_images_from_video(input_video_file, [0])
        data_io.write_images_to_file(bkg_image, output_bkg_image_path, file_name_prefix="lab_scene_bkg")
    
    print("Extract frame ids for each instance of a class")
    input_cvat_file=args.cvat_file
    class_data = data_io.parse_cvat_annotations(input_cvat_file)
    for iclass in class_data:
        img_ids = []
        rois = []
        for j in class_data[iclass]:
            img_ids.append(j[1])
            rois.append(j[2])
        if iclass == "Hand": 
            images = data_io.extract_images_from_video(input_video_file, img_ids)
        else:
            images = data_io.extract_images_from_video(input_video_file, img_ids, rois)
        od = os.path.join(args.class_data_image_path, iclass)
        data_io.write_images_to_file(images, od, file_name_prefix=f"{iclass}")
