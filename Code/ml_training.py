#from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import json
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import random
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
class Train_ai:
    def __init__(self):
        print("MODEL TRAINING")
        #self.model = self.make_model
        #print(self.model)

    def make_model_alternative(self):
        # Arquitetura modificada
        model = Sequential()
        model.add(Dense(units=512, input_dim=340, activation='relu'))  # Camada de entrada
        model.add(Dropout(0.5))  # Dropout para regularização
        model.add(Dense(256, activation='relu'))  # Camada oculta
        model.add(BatchNormalization())  # Normalização em lote para estabilizar o treinamento
        model.add(Dropout(0.5))  # Dropout para regularização
        model.add(Dense(128, activation='relu'))  # Camada oculta
        model.add(BatchNormalization())  # Normalização em lote
        model.add(Dropout(0.5))  # Dropout para regularização
        model.add(Dense(1, activation='sigmoid'))  # Camada de saída
        model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
        return model

    def make_model(self):
        # basic arquiteture
        model = Sequential()
        # model.add(tipo_camada(nNeuro,nInput,activeFunc)) INPUT LAYER
        model.add(Dense(units=340,input_dim=340,activation='relu'))
        # model.add(tipo_camada(nNeuro,activeFunc)) HIDEN LAYER
        model.add(Dense(240,activation='relu'))
        model.add(BatchNormalization())
        # model.add(tipo_camada(nNeuro,activeFunc)) HIDEN LAYER
        model.add(Dense(140,activation='relu'))
        model.add(BatchNormalization())
        # model.add(tipo_camada(nNeuro,activeFunc)) HIDEN LAYER
        model.add(Dense(40,activation='relu'))
        model.add(BatchNormalization())
        # model.add(tipo_camada(nNeuro,activeFunc)) OUTPUT LAYER
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
        return model


    def load_data(self,path_to_normal,path_to_abnormal,behavior=["normal","abnormal"]):
        x = []
        y = []
        normal = []
        abnormal = []
        num_examples = min(len(os.listdir(path_to_abnormal)),len(os.listdir(path_to_normal)))
        normal_folders = random.sample(os.listdir(path_to_normal),num_examples)
        abnormal_folders = random.sample(os.listdir(path_to_abnormal),num_examples)
        cont = 1
        for folder in normal_folders:
            print(f"Count: {cont} folder: {folder}")
            folder_path = os.path.join(path_to_normal, folder)
            if os.path.isdir(folder_path):
                normal_data = self.preprocess_data(folder_path,behavior[0])
                normal.extend(normal_data)
            cont+=1
        normal = np.array(normal)

        normal_data = {
            "normal": normal.tolist(),  # Convertendo o array numpy em uma lista Python
            "shape": normal.shape,
            "data_example": normal[0].tolist()  # Convertendo o primeiro elemento do array em uma lista Python
        }

        # Salvando os dados em um arquivo JSON
        with open('normal_data_2.json', 'w') as json_file:
            json.dump(normal_data, json_file)
        count=0
        for folder in abnormal_folders:
            print(f"Count: {cont} folder: {folder}")
            folder_path = os.path.join(path_to_abnormal, folder)
            if os.path.isdir(folder_path):
                abnormal_data = self.preprocess_data(folder_path,behavior[1])
                abnormal.extend(abnormal_data)
            cont+=1
        abnormal = np.array(abnormal)

        abnormal_data = {
            "abnormal": abnormal.tolist(),  # Convertendo o array numpy em uma lista Python
            "shape": abnormal.shape,
            "data_example": abnormal[0].tolist()  # Convertendo o primeiro elemento do array em uma lista Python
        }

        # Salvando os dados em um arquivo JSON
        with open('abnormal_data_2.json', 'w') as json_file:
            json.dump(abnormal_data, json_file)
                    

    def preprocess_data(self, path_file, behavior):
        if behavior == "normal":
            flag = 0
        elif behavior == "abnormal":
            flag = 1
        else:
            print("Behavior unreconized")
            return None  # Retorna None se o comportamento não for reconhecido

        frames_10 = []
        folder_path = os.path.join(path_file, "frames")

        if os.path.isdir(folder_path):
            lista_de_itens = os.listdir(folder_path)
            for i in range(len(lista_de_itens)-9):
                aux_key = []
                for j in range(i, i+10):
                    file_ = f"{folder_path}/frame_{j+1}.json"
                    with open(file_, 'r') as file:
                        data = json.load(file)
                        keypoints = data[0]['keypoints']
                        for x in range(len(keypoints)):
                            aux_key.append(keypoints[x][0])
                            aux_key.append(keypoints[x][1])

                aux_key.append(flag)
                frames_10.append(aux_key)

            # Converter para um array NumPy
            frames_10 = np.array(frames_10)

        return frames_10

                    
        
    def preparing_data(self,normal_folder="",abnormal_folder=""):
        with open(normal_folder,'r') as f:
            dados_normais = json.load(f)
        with open(abnormal_folder,'r') as f:
            dados_anormais = json.load(f)

        dados_normais = np.array(dados_normais['normal'])
        dados_anormais = np.array(dados_anormais['abnormal'])

        dados_combinados = np.concatenate((dados_normais, dados_anormais),axis=0)
        np.random.shuffle(dados_combinados)
        #print(dados_combinados)
        model_info = {
            "dados_combinados": dados_combinados.tolist(),
        }
        # with open("dados_combinados.json", "w") as json_file:
        #     json.dump(model_info, json_file, indent=4)
        x = dados_combinados[:,:-1]
        y = dados_combinados[:,-1]
        print(f"y: {y}")
        return x,y

    def train_model(self,normal_file="",abnormal_file=""):
        #self.x,self.y = self.preparing_data(normal_file,abnormal_file)
        self.x,self.y = self.preparing_data(normal_folder="./normal_data_2.json",abnormal_folder="./abnormal_data_2.json")
        scalar = MinMaxScaler()
        x_normalized  = scalar.fit_transform(self.x)
        x_train,x_test, y_train, y_test = train_test_split(x_normalized,self.y,test_size=0.2,random_state=42)
        model = self.make_model_alternative()
        trained_model = model.fit(x_train,y_train,epochs=50,batch_size=64,validation_data=(x_test,y_test))
        model.save('behavior_detection.h5')
        with open('historico_de_treinamento_2.json','w') as f:
            json.dump(trained_model.history,f)

    def test_model(self,model_file=""):
        model = tf.keras.models.load_model(model_file)
        x,y = self.preparing_data(normal_folder="./normal_data_2.json",abnormal_folder="./abnormal_data_2.json")
        scalar = MinMaxScaler()
        x_normalized  = scalar.fit_transform(x)
        x_train,x_test,y_train,y_test = train_test_split(x_normalized,y,test_size=0.2,random_state=42)
        print(f"y_test: {y_test}")
        print(f"Have 1:{np.isin(y_test,1.)} Have 0: {np.isin(y_test,0.)}")
        print(f"x_test: {x_test}")
        # Avaliar o modelo nos dados de teste
        loss, accuracy = model.evaluate(x_test, y_test)
        
        # Fazer previsões no conjunto de teste
        y_pred = np.argmax(model.predict(x_test), axis=1)
        print(f"y_pred: {y_pred}")
        # Calcular a matriz de confusão
        confusion = confusion_matrix(y_test, y_pred)
        self.save_confusion_matrix_image(confusion)
        # Exibir a matriz de confusão como um gráfico
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig("confusion_matrix.png")
        plt.close()

        # Exibir o relatório de classificação
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Extrair outras informações importantes sobre o modelo
        model_summary = model.summary()
        model_config = model.get_config()
        # Adicione mais informações conforme necessário

        # Salvar informações em um dicionário
        model_info = {
            "summary": model_summary,
            "config": model_config,
            "test_loss": loss,
            "test_accuracy": accuracy,
            "confusion_matrix": confusion.tolist(),
            "classification_report": classification_rep
        }

        # Salvar dicionário em um arquivo JSON
        with open("model_info.json", "w") as json_file:
            json.dump(model_info, json_file, indent=4)


    def save_confusion_matrix_image(self,confusion_matrix):
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(confusion_matrix))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig("./confusion_matrix_2.png")
        plt.close()

    def test_model_2(self):
        #x_test, y_test = test_data
        x,y = self.preparing_data(normal_folder="./normal_data_2.json",abnormal_folder="./abnormal_data_2.json")
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        scalar = MinMaxScaler()
        x_test_normalized = scalar.fit_transform(x_test)

        # Load the trained model
        model = tf.keras.models.load_model('./behavior_detection.h5')
        # Evaluate the model on the test data
        loss, accuracy = model.evaluate(x_test_normalized, y_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")

        # Make predictions
        y_pred = model.predict(x_test_normalized)

        # Convert predictions to binary classes
        y_pred_binary = np.round(y_pred).flatten()

        # Calculate confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred_binary)
        classification_rep = classification_report(y_test, y_pred_binary,output_dict=True)

        # Save model information to a dictionary
        model_info = {
            "summary": model.summary(),
            "config": model.get_config(),
            "test_loss": loss,
            "test_accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_rep
        }

        # Save dictionary to a JSON file
        with open("model_info_with_dropout.json", "w") as json_file:
            json.dump(model_info, json_file, indent=4)

        # Plot confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Save the plot as an image
        plt.savefig('confusion_matrix_dropout.png')

        # Show the plot
        plt.show()

if __name__=="__main__":
    print("Main")
    model = Train_ai()
    model.test_model_2()

