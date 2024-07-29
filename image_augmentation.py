import os
from PIL import Image, ImageEnhance
from tqdm import tqdm


NEW_FOLDERS = ['negatives', 'contrast', 'negatives_contrast']


class ImageAugmentation:

    def __init__(self, path, folder):
        self.path = path
        self.folder = folder
        self.create_folders()

    def create_folders(self):
        path = f'{self.path}/{self.folder}'
        for _name in NEW_FOLDERS:
            if not os.path.exists(f'{path}_{_name}'):
                os.makedirs(f'{path}_{_name}')

    def augmentation(self):
        alpha = 0.5
        for img in tqdm(os.listdir(f'{self.path}/{self.folder}')):
            image = Image.open(f'{self.path}/{self.folder}/{img}')

            negative = Image.eval(image, lambda x: 255 - x)
            negative.save(f'{self.path}/{self.folder}_{NEW_FOLDERS[0]}/{img}')

            enhancer_contrast = ImageEnhance.Contrast(image)
            contrast = enhancer_contrast.enhance(alpha)
            contrast.save(f'{self.path}/{self.folder}_{NEW_FOLDERS[1]}/{img}')

            enhancer_negative_contrast = ImageEnhance.Contrast(negative)
            negative_contrast = enhancer_negative_contrast.enhance(alpha)
            negative_contrast.save(f'{self.path}/{self.folder}_{NEW_FOLDERS[2]}/{img}')


if __name__ == "__main__":

    IA = ImageAugmentation('data/images_to_process', 'images')
    IA.augmentation()
