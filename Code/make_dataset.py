from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import json
import shutil

# mlp
# cnn



class Make_dataset:
    def __init__(self, model_path="yolov8m-pose.pt"):
        self.model = YOLO(model_path)
        self.model.to("cuda")
        self.anatomical_points = [
            [0, "N", "Nose"],
            [1, "LE", "Left Eye"],
            [2, "RE", "Right Eye"],
            [3, "LE", "Left Ear"],
            [4, "RE", "Right Ear"],
            [5, "LS", "Left Shoulder"],
            [6, "RS", "Right Shoulder"],
            [7, "LE", "Left Elbow"],
            [8, "RE", "Right Elbow"],
            [9, "LH", "Left Hand"],
            [10, "RH", "Right Hand"],
            [11, "LH", "Left Hip"],
            [12, "RH", "Right Hip"],
            [13, "LK", "Left Knee"],
            [14, "RK", "Right Knee"],
            [15, "LF", "Left Foot"],
            [16, "RF", "Right Foot"],
        ]

        self.cores = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#A52A2A", "#008000", "#800080", "#800000", "#008080", "#FF4500", "#4B0082", "#2E8B57", "#9370DB", "#DAA520",]
        self.skeleton_connections = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (0, 6),
            (0, 5),
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (12, 11),
            (12, 6),
            (11, 5),
            (12, 14),
            (14, 16),
            (11, 13),
            (13, 15),
        ]
    def process_video(self, video_path, output_folder="../video/",make_video=False,make_json=False,behavior=""):
        output_folder = output_folder+behavior
        os.makedirs(output_folder, exist_ok = True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_folder = os.path.join(output_folder, video_name)
        print(f"video_name:{video_name}")
        print(f"video_path: {video_path}")
        print(f"output_folder:{output_folder}")
        cap = cv2.VideoCapture(video_path)
        print(f"cap is Opened: {cap.isOpened()}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_filename = f"{video_name}.mp4"
        output_path_mp4 = os.path.join(output_video_folder, output_filename)
        if make_video:
            fourcc=cv2.VideoWriter_fourcc(*"XVID")
            out_mp4 = cv2.VideoWriter(output_path_mp4, fourcc, fps, (width, height))
        if make_json:
            output_json_folder = os.path.join(output_video_folder, "frames")
            os.makedirs(output_json_folder, exist_ok=True)
        position_x = []
        position_y = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #print(pil_image)
            results = self.model(pil_image, conf=0.3, show=False)
            #print(results)
            xList = [item for item in range(0,720)]
            if make_json:
                for i, r in enumerate(results):
                    d = 0
                    person_data = []
                    for person_keypoints, person_boxes in zip(r.keypoints.data, r.boxes.data):
                        key = person_keypoints.tolist()
                        flag_media = False
                        if flag_media:
                            x_mean, y_mean = 1,1
                            for i in range(len(key)):
                                x, y, conf = key[i]
                                x_mean += x
                                y_mean += y
                            mean = []
                            mean.append(x_mean/len(key))
                            mean.append(y_mean/len(key))
                        person_dict = {
                            "person_id": d,
                            "keypoints": key,
                            "box": person_boxes.tolist(),
                            "flag": behavior,
                        }
                        person_data.append(person_dict)
                        #x, y = r.keypoints.orig_shape
                        orig_shape = r.orig_shape
                        person_data.append(orig_shape)
                        if flag_media:
                            position_x.append(mean[0])
                            position_y.append(mean[1])
                        d+=1
                    #json_filename = f"pose_estimation_frame_{frame_count+1}_person_{i}.json"
                    json_filename = f"frame_{frame_count+1}.json"
                    json_filepath = os.path.join(output_json_folder, json_filename)
                    with open(json_filepath, "w") as json_file:
                        json.dump(person_data, json_file, indent=4)

            frame_count += 1
            if make_video:
                black_image = Image.new("RGB", (width, height),color="black")
                draw = ImageDraw.Draw(black_image)
                for r in results:
                    d = 0
                    person_data = []
                    for person_keypoints, person_boxes in zip(r.keypoints.data, r.boxes.data):
                        key = person_keypoints.tolist()
                        x_mean, y_mean = 1,1
                        i = 0
                        for point in person_keypoints:
                            x, y, conf = point.tolist()
                            x,y = int(x), int(y)
                            x_mean += x
                            y_mean += y
                            draw.ellipse([x-5,y-5,x+5,y+5], fill=self.cores[i])
                            i+=1
                        if person_keypoints.tolist():
                            i=0
                            for connection in self.skeleton_connections:
                                point1 = person_keypoints[connection[0]].tolist()
                                point2 = person_keypoints[connection[1]].tolist()
                                x1,y1,_ = map(int, point1)
                                x2,y2,_ = map(int, point2)
                                draw.line([(x1, y1), (x2, y2)], fill=self.cores[i%len(self.cores)])
                                i+=1
                frame_with_keypoints = cv2.cvtColor(np.array(black_image),cv2.COLOR_RGB2BGR)
                out_mp4.write(frame_with_keypoints)

        cap.release()
        if make_video:
            out_mp4.release()


    def video_rename(prefix, folder_path, folder_destiny):
        files = os.listdir(folder_path)
        #print(files)
        if not os.path.exists(folder_destiny):
            os.makedirs(folder_destiny)
        count = 1
        for file in files:
            if file.endswith(".mp4"):
                new_name = f"{prefix}{count}.mp4"
                os.rename(os.path.join(folder_path, file), os.path.join(folder_destiny,new_name))
                count+=1

    import os

    def video_rename_mp4s(self,folder_path, folder_destiny):
        files = os.listdir(folder_path)
        if not os.path.exists(folder_destiny):
            os.makedirs(folder_destiny)
        for file in files:
            if file.endswith(".mp4"):
                # Encontra o índice do primeiro espaço seguido de um número
                index = file.find(" ")
                if index != -1:
                    new_name = f"n{file[1:index]}.mp4"
                    os.rename(os.path.join(folder_path, file), os.path.join(folder_destiny, new_name))
                else:
                    print(f"O arquivo {file} não segue o padrão esperado e não foi renomeado.")

    def process_all_videos(self,folder_path,output_folder,make_video,make_json,behavior):
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(folder_path,file)
                self.process_video(video_path, output_folder, make_video=False, make_json=True,behavior=behavior)
                #self.process_video(video_path, output_folder, make_video=True, make_json=False,behavior=behavior)
    @staticmethod
    def extract_number(file_name):
        return int(file_name.split('_')[1].split('.')[0])

    def preprocessing_data(self, file_path, new_file_path):
        video_folder = os.path.join(file_path, "frames")
        video_folder_after = os.path.join(new_file_path, "frames")

        if not os.path.exists(video_folder_after):
            os.makedirs(video_folder_after)

        json_files = []
        for f in os.listdir(video_folder):
            if f.endswith(".json"):
                json_files.append(f)

        json_files = sorted(json_files, key=self.extract_number)

        for f in json_files:
            json_path = os.path.join(video_folder, f)
            json_after_path = os.path.join(video_folder_after, f)

            with open(json_path, 'r') as file:
                data = json.load(file)
                if data != []:
                    shutil.copy(json_path, json_after_path)
                            
    def preprocessing_data_all(self,folder_path="",folder_destiny="",flag="cleaning"):
        for f in os.listdir(folder_path):
            match flag:
                case "cleaning":
                    video_folder = os.path.join(folder_path, f)
                    new_video_folder = os.path.join(folder_destiny,f)
                    print(f"vFolder: {video_folder}\n nvFolder: {new_video_folder}")
                    print("-------------------------\n")
                    self.preprocessing_data(video_folder,new_video_folder)
                case "reindex":
                    video_folder = os.path.join(folder_path, f)
                    print(f"Processing folder -> {video_folder}")
                    self.reindex_jsons(video_folder)
                case _:
                    print(f"Unrecognized flag: {flag}")

    def reindex_jsons(self,folder_path):
        json_files = []
        folder_path = os.path.join(folder_path,"frames")
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith(".json"):
                    json_files.append(f)
            json_files = sorted(json_files, key=self.extract_number)

            for i in range(len(json_files)):
                old_name = os.path.join(folder_path,json_files[i])
                new_name = os.path.join(folder_path, f"frame_{i+1}.json")
                os.rename(old_name,new_name )
            
if __name__ == "__main__":
    video_processor = Make_dataset()
    #video_processor.process_all_videos("../videos/Normal/", "../videos/dataset/",False,True,"normal")
    #video_processor.process_all_videos("../videos/Abnormal/", "../videos/dataset/",False,True,"abnormal")
    #video_processor.preprocessing_data(file_path="../videos/dataset/normal/n1/",new_file_path="../videos/dataset2/normal/n1")
    #video_processor.preprocessing_data_all(folder_path="../videos/dataset/normal/",folder_destiny="../videos/dataset2/normal/")
    #video_processor.preprocessing_data_all(folder_path="../videos/dataset/abnormal/",folder_destiny="../videos/dataset2/abnormal/")
    video_processor.preprocessing_data_all(folder_path="../videos/dataset2/abnormal/",flag="reindex")























