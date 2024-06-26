import os
import yaml

from yolo_annotate import YoloLabeling, FolderManager
from labelImg_annotate import FileManager, launch_labelimg

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.dataset_path = config.get('base_root')
        self.yolo_model_path = config.get('yolo_weight')

        self.train_percentage = config.get('train_perc')
        self.test_percentage = config.get('test_perc')
        self.val_percentage = config.get('val_perc')

        self.class_file = config.get('class_file')

        if self.train_percentage + self.test_percentage + self.val_percentage != 1:
            raise ValueError("Sum of train, test and val percentage should be equal to 1")
        
def main():
    # Load and read config file
    config_path = 'config.yaml'
    config_loader = ConfigLoader(config_path)
    config_loader.load_config()

    # Create all the folders for the dataset
    folder_manager = FolderManager(config_loader)
    folder_manager.moove_images()
    #folder_manager.create_folders()

    # Run YOLO in inference and save the predicted labels as txt files
    yolo_labeling = YoloLabeling(config_loader, folder_manager)
    yolo_labeling.create_labels()

    # Attempt to run the LabelImg command
    labelimg_manager = FileManager(config_loader.dataset_path, config_loader.class_file) # Copy the annotations to the image folder for labelImg
    launch_labelimg(class_file=config_loader.class_file)
    # Move the annotations back to the original folder
    labelimg_manager.moove_labelimg_labels()
    folder_manager.create_folders()

if __name__ == '__main__':
    main()