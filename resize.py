import os
from PIL import Image
from multiprocessing import freeze_support

def resize_and_save_images(source_dir, target_dir, image_size=(64, 64)):
    """
    Resize all images in source_dir and save them to target_dir with the same directory structure.
    
    Args:
        source_dir (str): Path to the source directory (Train, Test, or Val).
        target_dir (str): Path to the target directory where resized images will be saved.
        image_size (tuple): Target size to resize images to (default: (64, 64)).
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Walk through the source directory and process each image
    for root, dirs, files in os.walk(source_dir):
        # Create the corresponding directory in target_dir
        relative_path = os.path.relpath(root, source_dir)
        target_subdir = os.path.join(target_dir, relative_path)
        
        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)
        
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                # Full file path in the source directory
                file_path = os.path.join(root, file)
                
                # Open and resize the image
                with Image.open(file_path) as img:
                    img_resized = img.resize(image_size, Image.Resampling.LANCZOS)
                    if img.mode in ['P', 'RGBA']:
                        img_resized = img_resized.convert('RGB')

                    # Save the resized image to the target directory
                    save_path = os.path.join(target_subdir, file)
                    img_resized.save(save_path)
                    print(f"Resized and saved {save_path}")


if __name__ == '__main__':
    freeze_support()
    # Paths to the source folders (Train, Test, Val)
    source_root = './data/split'  # Replace with the actual path
    target_root = './data/split224'   # Replace with the path where resized images will be saved

    # Resize and save images from Train, Test, and Val folders
    resize_and_save_images(os.path.join(source_root, 'Train'), os.path.join(target_root, 'Train'), image_size=(224, 224))
    resize_and_save_images(os.path.join(source_root, 'Test'), os.path.join(target_root, 'Test'), image_size=(224, 224))
    resize_and_save_images(os.path.join(source_root, 'Validate'), os.path.join(target_root, 'Validate'), image_size=(224, 224))
