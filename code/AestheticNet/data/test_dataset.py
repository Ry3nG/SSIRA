import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import warnings
import requests
from tqdm import tqdm  # For progress bar, optional
from bs4 import BeautifulSoup

PATH_AVA_TXT = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/AVA.txt"
PATH_AVA_IMAGE = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/image"

default_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class AVADataset(Dataset):
    """AVA dataset for Aesthetic Score Prediction"""

    def __init__(self, txt_file, root_dir, transform=default_transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(txt_file, delim_whitespace=True, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image_id = self.data_frame.iloc[idx, 1]
        img_name = os.path.join(self.root_dir, f"{image_id}.jpg")
        image = Image.open(img_name).convert("RGB")

        # compute the aesthetic score as the average of the ratings
        ratings = self.data_frame.iloc[idx, 2:12].values
        score = sum((i + 1) * ratings[i] for i in range(10)) / sum(ratings)

        if self.transform:
            image = self.transform(image)

        return image, score


# Other necessary imports and constants...

AVA_URL_FOR_ID = "http://www.dpchallenge.com/image.php?IMAGE_ID={}"


def extract_image_url(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    img_tags = soup.find_all("img")

    for img in img_tags:
        if img.get("src") and "images.dpchallenge.com/images_challenge/" in img["src"]:
            return img["src"]
    return None


def download_image(image_id, download_dir):
    page_url = AVA_URL_FOR_ID.format(image_id)
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(page_url, headers=headers)

    if response.status_code == 200 and "text/html" in response.headers.get(
        "content-type", ""
    ):
        image_url = extract_image_url(response.content)
        if image_url:
            image_response = requests.get(image_url, headers=headers)
            if image_response.status_code == 200:
                img_path = os.path.join(download_dir, f"{image_id}.jpg")
                with open(img_path, "wb") as file:
                    file.write(image_response.content)
                return True
    return False


def is_corrupted(img_path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(img_path) as img:
                img.verify()
        return False
    except Exception:
        return True


def check_and_download_images(dataset):
    for idx in range(len(dataset)):
        image_id = dataset.data_frame.iloc[idx, 1]
        img_path = os.path.join(dataset.root_dir, f"{image_id}.jpg")

        if not os.path.exists(img_path) or is_corrupted(img_path):
            print(
                f"Image {image_id} is missing or corrupted. Attempting to download..."
            )
            if download_image(image_id, dataset.root_dir):
                if not is_corrupted(img_path):
                    print(f"Image {image_id} successfully downloaded and verified.")
                else:
                    print(f"Downloaded image {image_id} is corrupted.")
            else:
                print(f"Failed to download image {image_id}.")


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
            print(f"Image {image_id} is verified.")
            score_list = dataset.data_frame.iloc[idx, 2:12].values
            score = sum((i + 1) * score_list[i] for i in range(10)) / sum(score_list)
            print(f"Image {image_id} has score {score}")
            
            # check abnormal score
            if score > 10 or score < 0:
                print(f"Image {image_id} has abnormal score {score}")
                corrupted_images.append(image_id)
            else:
                ok_images.append(image_id)
        
        print(f"Number of corrupted images: {len(corrupted_images)}")

        print(f"Number of verified images: {len(ok_images)}")

        print(f"Currupted images: {corrupted_images}")


        


def main():
    txt_file = PATH_AVA_TXT
    root_dir = PATH_AVA_IMAGE
    dataset = AVADataset(
        txt_file=txt_file, root_dir=root_dir, transform=default_transform
    )
    check_and_download_images(dataset)
    #check_image_integrity(dataset)


if __name__ == "__main__":
    main()
