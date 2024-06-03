import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
# import albumentations as A

def load_images_from_folder(folder):
    """ Load images from a given folder. """
    return [os.path.join(folder, filename) for filename in os.listdir(folder)]

def split_data(image_paths, train_size=0.8, random_state=42):
    """ Split the data into training and validation sets. """
    return train_test_split(image_paths, test_size=1-train_size, random_state=random_state)

def save_image(image, path):
    """ Save an image to a specified path, handling exceptions. """
    try:
        cv2.imwrite(path, image)
    except Exception as e:
        print(f"Error saving image to {path}: {e}")

def create_and_verify_output_dirs(train_dir, val_dir):
    """ Ensure output directories exist. """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

def process_images(image_paths, output_dir, process_function):
    """ Process and save images using a provided function. """
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_filename)
        processed_image = process_function(img)
        save_image(processed_image, output_path)

def simple_cv2_augmentation(image):
    """ Apply simple OpenCV-based augmentations to an image including rotation, flipping,
        brightness and contrast adjustment, adding noise, shearing, and cutout. """
    # Rotation
    angle = np.random.randint(-15, 15)
    rows, cols = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, matrix, (cols, rows))

    # Horizontal flipping
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)

    # Brightness and contrast adjustment
    alpha = 1.0 + np.random.uniform(-0.2, 0.2)  # Contrast control (0.8-1.2)
    beta = np.random.uniform(-30, 30)  # Brightness control (-30 to 30)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Adding Gaussian noise
    noise = np.random.normal(0, np.random.uniform(0, 20), image.shape).astype(np.uint8)
    image += noise

    # Shear transformation
    shear_factor = np.random.uniform(-0.2, 0.2)
    shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    image = cv2.warpAffine(image, shear_matrix, (cols, rows))

    # Cutoutdef simple_cv2_augmentation(image):
    """ Apply simple OpenCV-based augmentations to an image. """
    angle = np.random.randint(-15, 15)
    rows, cols = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, matrix, (cols, rows))

    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)

    alpha = 1.0 + np.random.uniform(-0.2, 0.2)
    beta = np.random.uniform(-30, 30)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    noise = np.random.normal(0, np.random.uniform(0, 20), image.shape).astype(np.uint8)
    return image + noise
    y1 = np.clip(y - height // 2, 0, rows)
    x2 = np.clip(x + width // 2, 0, cols)
    y2 = np.clip(y + height // 2, 0, rows)

    image[y1:y2, x1:x2] = 0  # Fill the rectangle with black color

    return image


def albumentations_augment_and_save(image, output_path):
    """ Apply Albumentations library augmentations and save the result. """
    transform = A.Compose([
        A.CLAHE(),
        A.RandomRotate90(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.HueSaturationValue()
    ])

    augmented_image = transform(image=image)['image']
    save_image(augmented_image, output_path)

def expand_dataset():
    # Setup paths and directories
    positive_class_dir = 'filtered-gdd-data/train/gun'
    output_train_dir = 'test_augment/novas_imgs/train/'
    output_val_dir = 'test_augment/novas_imgs/val/'
    create_and_verify_output_dirs(output_train_dir, output_val_dir)

    # Load images and split data
    positive_images = load_images_from_folder(positive_class_dir)
    train_images, val_images = split_data(positive_images)

    # Process training and validation images
    process_images(train_images, output_train_dir, simple_cv2_augmentation)
    process_images(val_images, output_val_dir, simple_cv2_augmentation)

    print("Data processing complete. Images are split and saved.")

if __name__ == "__main__":
    expand_dataset()



