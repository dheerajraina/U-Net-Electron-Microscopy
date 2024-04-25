from keras.utils import normalize
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import settings
import uuid
from sklearn.model_selection import train_test_split


class DatasetGenerator(object):
    def __init__(self, mask_directory, image_directory, seed=816, generate_data_from_scratch=False):
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.seed = seed
        self.image_dataset = []
        self.mask_dataset = []

        if generate_data_from_scratch:
            self.split_tiff(image_directory,
                            f"{settings.DATA_PATH}generated/images/img", settings.CROP_DIM)
            self.split_tiff(self.mask_directory,
                            f"{settings.DATA_PATH}generated/masks/msk", settings.CROP_DIM)

        self.image_directory = f"{settings.DATA_PATH}generated/images/"
        self.mask_directory = f"{settings.DATA_PATH}generated/masks/"

    def split_tiff(self, input_path, output_prefix, patch_size):
        with Image.open(input_path) as img:
            for i in range(img.n_frames):
                img.seek(1)
                self.create_patches_from_file(img, output_prefix, patch_size)

    def create_patches_from_file(self, img,  output_prefix, patch_size):
        width, height = img.size
        num_patches_vertical = height // patch_size
        num_patches_horizontal = width // patch_size
        counter = 0
        patches = []
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                box = (j, i, j + patch_size, i + patch_size)
                patch = img.crop(box)
                output_path = f"{output_prefix}_{uuid.uuid4()}.tiff"
                patch.save(output_path, format="tiff")
                counter += 1

    def read_images_from_source(self, directory_path):
        files = os.listdir(directory_path)
        file_list = []
        for i, file_name in enumerate(files):
            if (file_name.split('.')[1] == "tiff"):
                file = cv2.imread(directory_path+file_name, 0)
                file = Image.fromarray(file)
                file = file.resize((settings.CROP_DIM, settings.CROP_DIM))
                file_list.append(np.array(file))
        return file_list

    def get_dataset(self):
        """
        Return a dataset
        """

        self.image_dataset = self.read_images_from_source(self.image_directory)
        self.mask_dataset = self.read_images_from_source(self.mask_directory)
        self.image_dataset = np.expand_dims(
            normalize(np.array(self.image_dataset), axis=1), axis=-1)
        self.mask_dataset = np.expand_dims(
            (np.array(self.mask_dataset)), -1)/255  # since mask only has two colors either 0 or 255

        return self.image_dataset, self.mask_dataset

    def split_dataset(self, test_size, val_size, image_dataset, mask_dataset):
        X, X_test, y, y_test = train_test_split(
            image_dataset, mask_dataset, test_size=test_size, random_state=102)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=355)

        return X_train, y_train, X_val, y_val, X_test, y_test
