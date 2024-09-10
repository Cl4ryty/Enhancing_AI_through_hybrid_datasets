import os
import shutil

# Define original dataset directory and new dataset directory
original_data_dir = "../datasets/NZDL/valid"
new_data_dir = "../datasets/NZDL2/valid"

# List all class folders in the original dataset
class_folders = [folder for folder in os.listdir(original_data_dir) if
                 os.path.isdir(os.path.join(original_data_dir, folder))]

# Create new directories and copy images
for class_folder in class_folders:
    original_class_path = os.path.join(original_data_dir, class_folder,
                                       'original')  # Path to the original images
    new_class_path = os.path.join(new_data_dir, class_folder)

    # Create the new class directory if it doesn't exist
    os.makedirs(new_class_path, exist_ok=True)

    # Check if the original directory exists and copy contents
    if os.path.exists(original_class_path):
        for image_file in os.listdir(original_class_path):
            original_image_path = os.path.join(original_class_path, image_file)
            if os.path.isfile(original_image_path):  # Ensure it's a file
                shutil.copy(original_image_path, new_class_path)

print("Images copied successfully!")
