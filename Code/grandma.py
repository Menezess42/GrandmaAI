from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import json
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler


class Grandma:
    def __init__(self):
        self.track_key = defaultdict(lambda: [])
        self.track_box = defaultdict(lambda: [])
        print("Load project")

    def cv2_to_pil(self,frame):
        '''Converts the image from cv2 to pil'''
        return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    def pil_to_cv2(self,pil_image):
        '''Converts the image from pil to cv2'''
        return np.array(pil_image)

    def save_abnormal_behavior(self):
        #save report as json
        #save video slice
        pass

    def draw_text(self, image, text, position, color=( 0, 0, 255), font_path=None, font_size=22):
        draw = ImageDraw.Draw(image)
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default(font_size)
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle(bbox, fill="white")
        draw.text(position, text, fill=color, font=font,troke_width=5)
        return image

    def slide_window(self,data):
        try: 
            track_id = data[0].boxes.id.int().tolist()
            keypoints = data[0].keypoints.xy.cuda()
            boxes = data[0].boxes.xyxy.cuda()
            for box,kp,id in zip(boxes,keypoints,track_id):
                keypoint = self.track_key[id]
                bx = self.track_box[id]
                keypoint.append(kp.tolist())
                self.track_box[id] = box.tolist()
        except:
            pass

    def process_image(self,image):
        '''Make some notation direct in the frame(image)'''

        # [X] convert cv2 image to pil image
        pil_image = self.cv2_to_pil(frame=image)

        # [X] Pass the image through the yolov8m-pose model
        data = self.yolo_model.track(pil_image,persist=True,verbose=False)
        image = data[0].plot(kpt_line=False,kpt_radius=0,labels=True)
        image = Image.fromarray(image)
        # [X] Pass data to the function that handle the slide window
        if data[0]!=[]: 
            self.slide_window(data)

        ids = self.track_key.keys()

        for id in ids:
            keypoints = self.track_key[id]
            if len(keypoints) == 10:
                # Convertendo o vetor de vetores para um array NumPy
                array_vetor_de_vetores = np.array(keypoints)

                # Redimensionando o array para uma forma unidimensional
                matriz_2d = array_vetor_de_vetores.reshape(-1,34)

                # Normalizando os dados
                scalar = MinMaxScaler()
                x_normal = scalar.fit_transform(matriz_2d)
                x_normal = x_normal.reshape(1,-1)
                behavior = self.behavior_model(x_normal)        # Transformando o valor contínuo em uma previsão binária
                binary_prediction = 1 if behavior > 0.5 else 0

                # Agora você pode usar binary_prediction como a previsão binária
                if binary_prediction==1:
                    #print("\033[91mABNORMAL BEHAVIOR\033[0m")
                    bx = self.track_box[id]
                    #print(bx)
                    image = self.draw_text(image=image,text="ABNORMAL",position=(bx[0],bx[1]-42))

                #else:
                self.track_key[id].pop(0)
        
        # [X] Converts back to cv2 image
        frame_with_drawing = self.pil_to_cv2(image)
        image = frame_with_drawing
        return image
    
    def read_video(self,video_path=""):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error, can't open video file")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = self.process_image(image=frame)
            cv2.imshow('frame',image)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        

    def load_models(self,yolo_model_path="",behavior_model_path=""):
        if yolo_model_path=="":
            print("YOLO model path is emptiy or wrong")
        else:
            self.yolo_model = YOLO(yolo_model_path)
        if behavior_model_path=="":
            print("Behavior model path is emptiy or wrong")
        else:
            self.behavior_model = tf.keras.models.load_model(behavior_model_path)


if __name__ == "__main__":
    final_project = Grandma()
    yolo_model_path ="../Models/yolov8m-pose.pt"
    behavior_model_path = "../Models/behavior_detection.h5"
    final_project.load_models(yolo_model_path=yolo_model_path,behavior_model_path=behavior_model_path)
    #final_project.read_video(video_path="../Media/n10.mp4")
    #final_project.read_video(video_path="../Media/a42.mp4")
    final_project.read_video(video_path="../Media/a26.mp4")
    final_project.read_video(video_path="../Media/n113.mp4")
    #final_project.read_video(video_path="../Media/final_test.mp4")
