import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torch.utils.data import Dataset
import warnings
import requests
from tqdm import tqdm  # For progress bar, optional
from bs4 import BeautifulSoup
import cv2
# import Dataset
from data.dataset import AVADataset, default_transform


PATH_AVA_TXT = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/AVA.txt"
PATH_AVA_IMAGE = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/image"

# Other necessary imports and constants...

AVA_URL_FOR_ID = "http://www.dpchallenge.com/image.php?IMAGE_ID={}"



def extract_image_url(html_content,target_id):
    soup = BeautifulSoup(html_content, "html.parser")
    img_tags = soup.find_all("img")

    for img in img_tags:
        if img.get("src") and "images.dpchallenge.com/images_challenge/" in img["src"]:
            
            # also should end with image_id.jpg
            if img["src"].endswith(f"{target_id}.jpg"):
                return img["src"]
    return None



def is_corrupted(img_path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(img_path) as img:
                img.verify()
        return False
    except (OSError, UnidentifiedImageError) as e:
        print(f"Image {img_path} is corrupted because of {e}")
        return True
    except Exception as e:
        print(f"Image {img_path} is corrupted because of {e}")
        return True



def check_image_integrity(dataset):
    """
    loop through the dataset and check if the images are corrupted
    """
    corrupted_images = []
    ok_images = []

    for idx in range(len(dataset)):
        image_id = dataset.data_frame.iloc[idx, 1]
        img_path = os.path.join(dataset.root_dir, f"{image_id}.jpg")

        
        if not os.path.exists(img_path) or is_corrupted(img_path):
            print(f"Image {image_id} is missing or corrupted.")
            corrupted_images.append(image_id)

        else:
            #print(f"Image {image_id} is verified.")
            score_list = dataset.data_frame.iloc[idx, 2:12].values
            score = sum((i + 1) * score_list[i] for i in range(10)) / sum(score_list)
            #print(f"Image {image_id} has score {score}")
            
            # check abnormal score
            if score > 10 or score < 0:
                print(f"Image {image_id} has abnormal score {score}")
                corrupted_images.append(image_id)
            else:
                ok_images.append(image_id)
        
        #print(f"Number of corrupted images: {len(corrupted_images)}")

        #print(f"Number of verified images: {len(ok_images)}")

        #print(f"Currupted images: {corrupted_images}")

    return corrupted_images, ok_images

def is_corrupted_opencv(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image {img_path} is corrupted because of None")
            return True
        else:
            return False
    except Exception as e:
        print(f"Image {img_path} is corrupted because of {e}")
        return True
    
def check_image_integrity_opencv(dataset):
    """
    loop through the dataset and check if the images are corrupted
    """
    corrupted_images = []
    ok_images = []

    for idx in range(len(dataset)):
        image_id = dataset.data_frame.iloc[idx, 1]
        img_path = os.path.join(dataset.root_dir, f"{image_id}.jpg")

        
        if not os.path.exists(img_path) or is_corrupted_opencv(img_path):
            print(f"Image {image_id} is missing or corrupted.")
            corrupted_images.append(image_id)

        else:
            #print(f"Image {image_id} is verified.")
            score_list = dataset.data_frame.iloc[idx, 2:12].values
            score = sum((i + 1) * score_list[i] for i in range(10)) / sum(score_list)
            #print(f"Image {image_id} has score {score}")
            
            # check abnormal score
            if score > 10 or score < 0:
                print(f"Image {image_id} has abnormal score {score}")
                corrupted_images.append(image_id)
            else:
                ok_images.append(image_id)
        
        #print(f"Number of corrupted images: {len(corrupted_images)}")

        #print(f"Number of verified images: {len(ok_images)}")

        #print(f"Currupted images: {corrupted_images}")

    return corrupted_images, ok_images
   
def get_image_download_direct_url(image_id):
    page_url = AVA_URL_FOR_ID.format(image_id)
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(page_url, headers=headers)

    if response.status_code == 200 and "text/html" in response.headers.get(
        "content-type", ""
    ):
        image_url = extract_image_url(response.content, image_id)
        if image_url:
            return image_url
    return None

def download_image_to_path(image_id, target_path):
    image_url = get_image_download_direct_url(image_id)
    if image_url:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.content)
            return True
    return False

def check_image_integrity_from_list(dataset, list_jpgl):
    # check the ids listed in the given .jpgl file are ok
    corrupted_images = []
    ok_images = []
    ids = []
    with open(list_jpgl, "r") as f:
        ids = f.readlines()
    ids = [int(id.strip()) for id in ids]
    for idx in range(len(dataset)):
        image_id = dataset.data_frame.iloc[idx, 1]
        if image_id in ids:
            img_path = os.path.join(dataset.root_dir, f"{image_id}.jpg")
            if not os.path.exists(img_path) or is_corrupted(img_path):
                print(f"Image {image_id} is missing or corrupted.")
                corrupted_images.append(image_id)

            else:
                #print(f"Image {image_id} is verified.")
                score_list = dataset.data_frame.iloc[idx, 2:12].values
                score = sum((i + 1) * score_list[i] for i in range(10)) / sum(score_list)
                #print(f"Image {image_id} has score {score}")
                
                # check abnormal score
                if score > 10 or score < 0:
                    print(f"Image {image_id} has abnormal score {score}")
                    corrupted_images.append(image_id)
                else:
                    ok_images.append(image_id)
        
        #print(f"Number of corrupted images: {len(corrupted_images)}")

        #print(f"Number of verified images: {len(ok_images)}")

        #print(f"Currupted images: {corrupted_images}")
    return corrupted_images, ok_images
    

def main():
    custom_transform_options = [24]
    dataset =  AVADataset(
        txt_file=PATH_AVA_TXT,
        root_dir=PATH_AVA_IMAGE,
        custom_transform_options=custom_transform_options,
        ids=None,
        include_ids=False,
    )


    """
    corrupted_images, ok_images = check_image_integrity(dataset)

    print(f"Number of corrupted images: {len(corrupted_images)}")   
    print(f"Number of verified images: {len(ok_images)}")
    print(f"Currupted images: {corrupted_images}")
    """ 
    #Currupted_images: [2129, 502377, 532055, 570175, 501064, 564093, 593733, 627334, 639811, 397289, 499068, 501015, 556798, 523555, 11066, 564307, 547917, 1617, 512522]

    print("Integrity check for generic_ls_train.jpgl ---------------------------------")
    generic_ls_train_jpgl = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/aesthetics_image_lists/generic_ls_train.jpgl"
    corrupted_images, ok = check_image_integrity_from_list(dataset, generic_ls_train_jpgl)
    # exclude the corrupted images from the list, and save the new list to generic_ls_train_clean.jpgl
    generic_ls_train_clean_jpgl = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/aesthetics_image_lists/generic_ls_train_clean.jpgl"
    with open(generic_ls_train_jpgl, "r") as f:
        ids = f.readlines()
    ids = [int(id.strip()) for id in ids]

    with open(generic_ls_train_clean_jpgl, "w") as f:
        for id in ids:
            if id not in corrupted_images:
                f.write(f"{id}\n")

    



    print("Integrity check for generic_test.jpgl ---------------------------------")
    generic_test_jpgl = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/aesthetics_image_lists/generic_test.jpgl"
    corrupted_images, ok = check_image_integrity_from_list(dataset, generic_test_jpgl)
    # exclude the corrupted images from the list, and save the new list to generic_ls_train_clean.jpgl
    generic_test_clean_jpgl = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/aesthetics_image_lists/generic_test_clean.jpgl"
    with open(generic_test_jpgl, "r") as f:
        ids = f.readlines()
    ids = [int(id.strip()) for id in ids]

    with open(generic_test_clean_jpgl, "w") as f:
        for id in ids:
            if id not in corrupted_images:
                f.write(f"{id}\n")







    






if __name__ == "__main__":
    main()
