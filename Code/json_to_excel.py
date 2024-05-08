import pandas as pd
import json
import os

# Função para processar um único JSON e retornar um DataFrame
def process_json(json_data):
    rows = []
    for id_, data in json_data["track_data"].items():
        slide_window_values = []
        for slide_window, behavior in data["moda_behavior"].items():
            if behavior["Abnormal"] > behavior["Normal"]:
                slide_window_values.append((slide_window, 1))
            else:
                slide_window_values.append((slide_window, 0))
        row = {
            "video": json_data["video"],
            "frame_count": json_data["frame_count"],
            "ID": id_,
            "Abnormal": data["Abnormal"],
            "Normal": data["Normal"],
            "slideWindow_count": data["count_slideWindow_pass"],
        }
        row.update({
            slideWindow: v for slideWindow, v in slide_window_values
        })
        rows.append(row)
    return pd.DataFrame(rows)


# Processar todos os arquivos JSON em um diretório e salvar em arquivos Excel individuais no mesmo diretório
def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as file:
                json_data = json.load(file)
                df = process_json(json_data)
                output_filename = os.path.join(directory, os.path.splitext(filename)[0] + ".xlsx")
                df.to_excel(output_filename, index=False)


# Diretório contendo os arquivos JSON
directory = "../Reports_data/"

# Processar os arquivos JSON e criar arquivos Excel individuais
process_directory(directory)

