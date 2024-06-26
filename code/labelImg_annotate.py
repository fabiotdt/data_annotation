import os
import subprocess
import shutil

class FileManager:
    def __init__(self, dataset_path, class_path):
        
        self.dataset_path = dataset_path
        
        self.image = os.path.join(self.dataset_path, 'images')
        self.yolo_annotation = os.path.join(self.dataset_path, 'yolo_labels')
        self.labelimg_annotation = os.path.join(self.dataset_path, 'labels')

        self.class_file = class_path

        self.copy_annotations()

    def copy_annotations(self):

        # labelImg requiers that annotation and image files are in the same direcotry

        folders = os.listdir(self.yolo_annotation)
        for folder in folders:
            files = os.listdir(os.path.join(self.yolo_annotation))
            for file in files:
                if file.endswith('.txt'):
                    shutil.copy(os.path.join(self.yolo_annotation, file), os.path.join(self.image, file))

        shutil.copy(os.path.join(self.class_file, 'classes.txt'), os.path.join(self.image, 'classes.txt'))
    
    def moove_labelimg_labels(self):
        # Move the annotations back to the original folder
        files = os.listdir(self.image)
        for file in files:
            if file.endswith('.txt'):
                if file == 'classes.txt':
                    shutil.copy(os.path.join(self.image, file), os.path.join(self.dataset_path, 'classes_new.txt'))
                else:
                    os.rename(os.path.join(self.image, file), os.path.join(self.labelimg_annotation, file))


def launch_labelimg(image_path=None, class_file=None):
    try:
        # Prepare the command with the necessary arguments
        command = ["labelImg"]
        if image_path:
            command.append(image_path)
        if class_file:
            command.append(class_file)
        
        # Attempt to run the LabelImg command
        subprocess.run(command, check=True)
    except FileNotFoundError:
        # If the above fails, attempt to call it as a Python module
        try:
            command = ["python", "-m", "labelImg"]
            if image_path:
                command.append(image_path)
            if class_file:
                command.append(class_file)
            
            subprocess.run(command, check=True)
        except FileNotFoundError:
            print("LabelImg is not installed or not found in the PATH.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while trying to launch LabelImg: {e}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to launch LabelImg: {e}")

def main():

    dataset_path = 'dataset'
    FileManager(dataset_path)
    launch_labelimg()

if __name__ == '__main__':
    main()




