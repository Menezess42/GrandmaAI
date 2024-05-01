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
import math


class Grandma:
    def __init__(self):
        self.track_key = defaultdict(lambda: [])
        self.track_box = defaultdict(lambda: [])
        self.track_behavior = defaultdict(lambda: {"id":0,"1":0,"0":0,"inFrame_count": 0,"count_slideWindow_pass": 0})
        self.count_frames = 0
        print("Load project")

    def rest_not_0(valor):
        if valor == 0:
            return 1  # Para evitar divisão por zero
        num_digitos = math.floor(math.log10(abs(valor))) + 1  # Número de dígitos
        divisor = 10 ** (num_digitos - 1)  # 10 elevado ao (número de dígitos - 1)
        if valor%divisor == 0:
            return True
        return False

    def cv2_to_pil(self,frame):
        '''Converts the image from cv2 to pil'''
        return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    def pil_to_cv2(self,pil_image):
        '''Converts the image from pil to cv2'''
        return np.array(pil_image)

    def draw_text(self, image, text, position, color=( 0, 0, 255), font_path=None, font_size=22):
        if not isinstance(image, ImageDraw.ImageDraw):
            draw = ImageDraw.Draw(image)
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default(font_size)
        bbox = draw.textbbox(position, text, font=font)
        if text == "ABNORMAL":
            draw.rectangle(bbox, fill="white")
        else:
            draw.rectangle(bbox, fill="blue")
            color=(0,0,0)
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

        pil_image = self.cv2_to_pil(frame=image)

        data = self.yolo_model.track(pil_image,persist=True,verbose=False)
        image = data[0].plot(kpt_line=False,kpt_radius=0,labels=True)
        image = Image.fromarray(image)
        if data[0]!=[] or data[0]==data[0]: 
            self.slide_window(data)

        ids = self.track_key.keys()
        if ids:
            self.count_frames+=1
        window_count=""
        abnormal_b=""
        normal_b=""
        for id in ids:
            track_behavior=self.track_behavior[id]
            track_behavior["id"]=id
            track_behavior["inFrame_count"] +=1
            keypoints = self.track_key[id]
            window_count+=f" |{track_behavior['id']}: {track_behavior['count_slideWindow_pass']}"
            abnormal_b+=f" |{track_behavior['id']}: {track_behavior['1']}"
            normal_b+=f" |{track_behavior['id']}: {track_behavior['0']}"
            if len(keypoints) == 10:
                track_behavior["count_slideWindow_pass"]+=1
                array_vetor_de_vetores = np.array(keypoints)

                matriz_2d = array_vetor_de_vetores.reshape(-1,34)

                scalar = MinMaxScaler()
                x_normal = scalar.fit_transform(matriz_2d)
                x_normal = x_normal.reshape(1,-1)
                behavior = self.behavior_model(x_normal)        # Transformando o valor contínuo em uma previsão binária
                binary_prediction = 1 if behavior > 0.5 else 0

                if binary_prediction==1:
                    bx = self.track_box[id]
                    track_behavior["1"]+=1
                    image = self.draw_text(image=image,text="ABNORMAL",position=(bx[0],bx[1]-42))

                elif binary_prediction==0:
                    bx = self.track_box[id]
                    track_behavior["0"]+=1
                    image = self.draw_text(image=image,text="NORMAL",position=(bx[0],bx[1]-42))
                #if track_behavior['count_slideWindow_pass']%10==0:

                self.track_key[id].pop(0)
        
        # [X] Converts back to cv2 image
        draw_text = window_count+"\n"+abnormal_b+"\n"+normal_b
        image = self.draw_text(image=image, text=draw_text,position=(0,0))
        frame_with_drawing = self.pil_to_cv2(image)
        image = frame_with_drawing
        return image
    
    def read_video(self,video_path=""):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error, can't open video file")
        frame_by_frame = False
        while cap.isOpened():
            if not frame_by_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                image = self.process_image(image=frame)
                cv2.imshow('frame',image)
            else:
                key = cv2.waitKey(0)  # Espera infinita até que uma tecla seja pressionada

                if key & 0xFF == ord(' '):  # Se a tecla espaço for pressionada
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image = self.process_image(image=frame)
                    cv2.imshow('frame', image)
                elif key & 0xFF == ord('q'):  # Se a tecla 'q' for pressionada
                    break

            key = cv2.waitKey(25)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('t'):  # Se a tecla 'T' for pressionada
                frame_by_frame = not frame_by_frame  # Ativa/desativa o modo frame por frame


        cap.release()
        cv2.destroyAllWindows()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        info_to_save = {
            "video": f"{video_name}",
            "frame_count": self.count_frames,
            "track_data": self.track_behavior
        }
        flag = False
        if flag:
            json_file_path = os.path.join("../Reports_data/",f"{video_name}.json")
            with open(json_file_path, "w") as json_file:
                json.dump(info_to_save,json_file)
        else:
            print("Não entrou na flag")
        

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
    def run(video_path=""):
        final_project = Grandma()
        yolo_model_path ="../Models/yolov8m-pose.pt"
        behavior_model_path = "../Models/behavior_detection.h5"
        final_project.load_models(yolo_model_path=yolo_model_path,behavior_model_path=behavior_model_path)
        #final_project.read_video(video_path="path/to/the/video")
        final_project.read_video(video_path=video_path)
    # TEST 1
    for i in rage(3):
        run(video_path="../Media/test1_down_3act_3n.mp4")
    # TEST 2
    #for i in rage(3):
        #run(video_path="../Media/test4_up_3act_3n.mp4")
    # TEST 3
    #for i in rage(3):
        #run(video_path="../Media/test5_down_3act_1a_fence.mp4")
    # TEST 4
    #for i in rage(3):
        #run(video_path="../Media/test7_up_3act_1a_gate.mp4")
    # TEST 5
    #for i in rage(3):
        #run(video_path="../Media/test8_down_3act_2a_fence.mp4")
    # TEST 6
    #for i in rage(3):
        #run(video_path="../Media/test9_up_3act_2a_gate.mp4")
    # TEST 7
    #for i in rage(3):
        #run(video_path="../Media/test11_down_3act_3a_fence.mp4")
    # TEST 8
    #for i in rage(3):
        #run(video_path="../Media/test12_up_3act_3a_gate.mp4")
    # TEST 9
    #for i in rage(3):
        #run(video_path="../Media/test13_up_3act_1a_fence.mp4")
    # TEST 10
    #for i in rage(3):
        #run(video_path="../Media/test14_up_3act_2a_fence.mp4")
    # TEST 11
    #for i in rage(3):
        #run(video_path="../Media/test15_up_3act_3a_fence.mp4")
