from ultralytics import YOLO
import os
from tqdm import tqdm
import random
        
class FolderManager:
    def __init__(self, config_info):
        
        self.config_info = config_info
        self.dataset_path = self.config_info.dataset_path

        self.image = os.path.join(self.dataset_path, 'images')
        os.makedirs(self.image, exist_ok=True)
        self.yolo_annotation = os.path.join(self.dataset_path, 'yolo_labels')
        os.makedirs(self.yolo_annotation, exist_ok=True)
        self.annotation = os.path.join(self.dataset_path, 'labels')
        os.makedirs(self.annotation, exist_ok=True)
 
    def create_folders(self):

        self.image_train = os.path.join(self.image,'train')
        os.makedirs(self.image_train, exist_ok=True)
        self.annotation_train = os.path.join(self.annotation,'train')
        os.makedirs(self.annotation_train, exist_ok=True)

        self.image_test = os.path.join(self.image,'test')
        os.makedirs(self.image_test, exist_ok=True)
        self.annotation_test = os.path.join(self.annotation,'test')
        os.makedirs(self.annotation_test, exist_ok=True)

        self.image_val = os.path.join(self.image,'val')
        os.makedirs(self.image_val, exist_ok=True)
        self.annotation_val = os.path.join(self.annotation,'val')
        os.makedirs(self.annotation_val, exist_ok=True)

        self.divide_dataset()

    def moove_images(self):
        images = os.listdir(self.dataset_path)

        for image in images:
            if image.endswith('.jpg'):
                os.rename(os.path.join(self.dataset_path, image), os.path.join(self.image, image))

    def divide_dataset(self):
        
        files_list= os.listdir(self.image)
        files = [file for file in files_list if file.endswith('.jpg')]
        random.shuffle(files) # Randomize the files
        
        train_files = files[:int(len(files) * self.config_info.train_percentage)]
        test_files = files[int(len(files) * self.config_info.train_percentage):int(len(files) * (self.config_info.train_percentage + self.config_info.test_percentage))]
        val_files = files[int(len(files) * (self.config_info.train_percentage + self.config_info.test_percentage)):]

        for file in train_files:
            os.rename(os.path.join(self.image, file), os.path.join(self.image_train, file))
            os.rename(os.path.join(self.annotation, file.replace('.jpg', '.txt')), os.path.join(self.annotation_train, file.replace('.jpg', '.txt')))
        for file in test_files:
            os.rename(os.path.join(self.image, file), os.path.join(self.image_test, file))
            os.rename(os.path.join(self.annotation, file.replace('.jpg', '.txt')), os.path.join(self.annotation_test, file.replace('.jpg', '.txt')))
        for file in val_files:
            os.rename(os.path.join(self.image, file), os.path.join(self.image_val, file))
            os.rename(os.path.join(self.annotation, file.replace('.jpg', '.txt')), os.path.join(self.annotation_val, file.replace('.jpg', '.txt')))

class YoloLabeling:
    def __init__(self, config_info, folder_manager, conf=0.75):
        self.config_info = config_info
        self.folder_manager = folder_manager

        self.yolo = YOLO(self.config_info.yolo_model_path)
        self.conf = conf

    def create_labels(self):

        for file in tqdm(os.listdir(self.folder_manager.image)):
            if file.endswith('.jpg'):
                annotation_file = file.replace('.jpg', '.txt')

            results = self.yolo.predict(os.path.join(self.folder_manager.image, file), show=False, conf=self.conf)

            self.process_results(results, self.folder_manager.yolo_annotation, annotation_file)
    
    def process_results(self, results, annotation_folder, annotation_file):
        # Create an empty list to store the annotations
        annotations = []

        # Iterate over the predicted results
        for result in results[0]:
            bbox = result.boxes.xywhn[0].cpu().numpy()

            annotations.append(bbox)

        # Save the coco data to a txt file
        with open(os.path.join(annotation_folder, annotation_file), 'w') as f:
            for annotation in annotations:
                f.write('0 ' + ' '.join(str(element) for element in annotation) + '\n')
                
def main():
    pass
    
if __name__ == '__main__':
    main()






