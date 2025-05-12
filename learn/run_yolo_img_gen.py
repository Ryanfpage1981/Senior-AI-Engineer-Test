import argparse
import cv2
import numpy as np
import os
from data_tools import data_augmentation, data_io
"""
App to run through and create a dataset for training a YOLO model. Use agumentation to build up a training images. Splits by label number and computes the number of images that are required to get to that split
"""


def extract_number(filename):
    """
    Matching to get ordered files using postfix
    """
    import re
    match = re.search(r'_(\d+)(?:\.\w+)?$', filename)
    if match:
        return int(match.group(1))
    return 0  # Default value if no number is found

def create_random_array_with_max_sum(size, min_val, max_val, max_sum):
    """
    Creates an array of random numbers between min_val and max_val (inclusive),
    ensuring the total sum doesn't exceed a specified maximum value.
    """
    array = np.zeros(size, dtype=int) 
    current_sum = 0
    
    # Fill the array with random numbers
    for i in range(size):
        remaining_allowance = max_sum - current_sum
        
        if remaining_allowance <= 0:
            return array
        
        upper_bound = min(max_val, remaining_allowance)
        
        random_value = np.random.randint(min_val, upper_bound + 1)
        
        array[i] = random_value
        current_sum += random_value
    
    return array

def setup_yolo_dir_structure(path_name):
    
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    dirs=[]    
    dirs.append(os.path.join(path_name, 'train'))
    dirs.append(os.path.join(path_name, 'val'))
    dirs.append(os.path.join(dirs[0], 'images'))
    dirs.append(os.path.join(dirs[0], 'labels'))

    dirs.append(os.path.join(dirs[1], 'images'))
    dirs.append(os.path.join(dirs[1], 'labels'))

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    return dirs[2:]

def generate_hand_dataset(class_data, class_imgs_path, class_id): 
    """
    Build up dataset for the hand class, this is a special case as we dont use ROIs and
    image generation
    """
    imgs = []
    labels = [] 
    hand_files = [f for f in os.listdir(class_imgs_path) if os.path.isfile(os.path.join(class_imgs_path, f))]
    hand_files = sorted(hand_files, key=extract_number) 
    
    print(f"Number of Hand Samples {len(hand_files)}")
    n_labels=0
    for i,j in zip(hand_files, class_data):
        imgs.append(cv2.imread(os.path.join(class_imgs_path,i)))
        img_label = [] ## Make consistent with other classes that can have more than one instance per image
        img_label.append([class_id,j[2]])
        labels.append(img_label)
        n_labels += 1
    return imgs, labels, n_labels


def generate_syn_dataset(class_data, class_imgs_path, n_samples_in_img, class_id, bkg_img):
    """
    Build a dataset with using crops from original images and move them onto a blank scene

    TODO support more than a single class per image
    """

    imgs = []
    labels = []
    num_original_samples = len(class_data)
    sample_idx=0
    roi_files = [f for f in os.listdir(class_imgs_path) if os.path.isfile(os.path.join(class_imgs_path, f))] ## Get files
    roi_files = sorted(roi_files, key=extract_number) ## Order files
    roi_files = [os.path.join(class_imgs_path, f) for f in roi_files] ##add path back ##TODO redo data storage 
    step_size_x = class_data[0][2][2] - class_data[0][2][0] // 2 ##approx image x of box ##TODO better tracking of size and placement
    step_size_y = class_data[0][2][3] - class_data[0][2][1] // 2 ##approx image y of box ##TODO better tracking of size and placement
    n_labels=0
    for i in n_samples_in_img:
        print(f"Number of classes in image {i}")
        if i > 0:
            img = bkg_img.copy()
            img_labels = []
            for j in range(i):
                epsilon = np.random.randint(0, 50) ##Random jitter from original location
                if num_original_samples > 1:
                    sample_idx = np.random.randint(0, num_original_samples-1) ##Sample the examples of the class we have TODO Sample better
                print(f"Sample {sample_idx}")
                orig_pos = class_data[sample_idx][2]
                random_step_x = np.random.randint(0, step_size_x) ##TODO Experiement with +/- shiting
                random_step_y = np.random.randint(0, step_size_y)
                new_centre = [orig_pos[2] - orig_pos[0] // 2 + epsilon + random_step_x*j , orig_pos[3] - orig_pos[1] // 2 + epsilon + random_step_y*j] 
                print(f"Oringal pos {orig_pos}")
                try:
                    img, bbox = data_augmentation.compositor(img, cv2.imread(roi_files[sample_idx]), new_centre)
                    img_labels.append([class_id, bbox])
                    n_labels += 1
                except Exception as e:
                    print("Error generating image ignore this sample") ##This happens if we go out of bounds currently need to fix in the compositor
                    pass
            labels.append(img_labels)
            imgs.append(img.copy())
    return imgs, labels, n_labels

def split_dataset_by_n_labels(labels, n_labels, n_imgs, frac_t=0.9):
    """
    Split the dataset by labels not by image number so need to check how many
    are in each image to make the split
    """
    if n_labels == n_imgs:
        train_s = int(np.floor(n_labels * 0.9))
        valid_s = n_imgs - train_s
    else:
        count = 0
        target = int(np.floor(n_labels * 0.9))
        print(target)
        for i, im_l in enumerate(labels):
            count += len(im_l)
            if count > target:
                print(count)
                train_s = i - 1 ##go back one image
                break
        valid_s = n_imgs - train_s
    return train_s, valid_s



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bkg_image", default="data/original/bkg_images/lab_scene_bkg_0.png")
    parser.add_argument("--cvat_file", default="data/cvat_annotations/labels_run_0_2000.xml")
    parser.add_argument("--class_data_image_path", default="data/original")
    parser.add_argument("--n_training_images", default=60) 
    parser.add_argument("--n_class_labels", default=130) ##This is going to create an unbalanced dataset - with hands 
    parser.add_argument("--yolo_data_dir", default="data/yolo")
    parser.add_argument("--training_frac", default=0.9)
    args = parser.parse_args()
    classmapping = {"Hand": 0, "Bottle": 1, "PetriDish": 2}
    
    print("Lets create a new dataset for Yolo")
    
    class_data = data_io.parse_cvat_annotations(args.cvat_file) 
    ds_loc = setup_yolo_dir_structure(args.yolo_data_dir) 
    ##hands I have only used the original images and bbox, would extend augmentation further in future
    imgs_h, labels_h, n_labels_h = generate_hand_dataset(class_data['Hand'], os.path.join(args.class_data_image_path,'Hand'), classmapping['Hand'])
    
    ##create an augmented data set of bottle and petri dish - Does not do anything with lighting or viewing angle. 
    n_training_images = args.n_training_images ##number of images
    n_training_labels = args.n_class_labels ## balance across classes and images
    class_samples = {}
    max_pd_per_image = 5
    min_pd_per_image = 1
    dish_samples_per_image = create_random_array_with_max_sum(n_training_images, min_pd_per_image, max_pd_per_image, n_training_labels) 
    np.random.shuffle(dish_samples_per_image) ##shuffle in case we have loads of 0 at the end if we reach max early
    
    max_b_per_image = 2
    min_b_per_image = 1
    bottle_samples_per_image = create_random_array_with_max_sum(n_training_images, min_b_per_image, max_b_per_image, n_training_labels)  
    np.random.shuffle(bottle_samples_per_image)
    
    
    imgs_b, labels_b, n_labels_b = generate_syn_dataset(class_data['Bottle'], os.path.join(args.class_data_image_path,'Bottle'), bottle_samples_per_image, classmapping['Bottle'], cv2.imread(args.bkg_image))
    
    imgs_pd, labels_pd, n_labels_pd = generate_syn_dataset(class_data['PetriDish'], os.path.join(args.class_data_image_path,'PetriDish'), dish_samples_per_image, classmapping['PetriDish'], cv2.imread(args.bkg_image))

   
    ##split each dataset into train and validation before writing
    ##easy for hands
    h_t , h_v = split_dataset_by_n_labels(labels_h, n_labels_h, len(imgs_h), frac_t=args.training_frac) 
    b_t , b_v = split_dataset_by_n_labels(labels_b, n_labels_b, len(imgs_b), frac_t=args.training_frac) 
    pd_t , pd_v = split_dataset_by_n_labels(labels_pd, n_labels_pd, len(imgs_pd), frac_t=args.training_frac) 
    #Aim for a good balance on the class labels
    print("############## Data Stats ##############")
    print(f"Hand data labels : {n_labels_h}")
    print(f"Hand data imgs : {len(imgs_h)}")
    print(f"Bottle data labels : {n_labels_b}")
    print(f"Bottle data imgs : {len(imgs_b)}")
    print(f"PetriDish data labels : {n_labels_pd}")
    print(f"PetriDish data imgs : {len(imgs_pd)}")
    print(f"Hand ts : {h_t}")
    print(f"Hand vs : {h_v}")
    print(f"Bottle ts : {b_t}")
    print(f"Bottle vs : {b_v}")
    print(f"PetriDish ts : {pd_t}")
    print(f"PetriDish vs : {pd_v}")
     
    combined_ts_imgs = imgs_h[:h_t] + imgs_b[:b_t] + imgs_pd[:pd_t]
    combined_ts_labels = labels_h[:h_t] + labels_b[:b_t] + labels_pd[:pd_t]
    combined_vs_imgs = imgs_h[h_t:] + imgs_b[b_t:] + imgs_pd[pd_t:]
    combined_vs_labels = labels_h[h_t:] + labels_b[b_t:] + labels_pd[pd_t:]

    data_io.write_images_to_file(combined_ts_imgs, ds_loc[0]) 
    data_io.write_images_to_file(combined_vs_imgs, ds_loc[2]) 
   
    
    data_io.write_labels_to_file(combined_ts_labels, ds_loc[1], convert_from_cvat_to_yolo=True)
    data_io.write_labels_to_file(combined_vs_labels, ds_loc[3], convert_from_cvat_to_yolo=True)

