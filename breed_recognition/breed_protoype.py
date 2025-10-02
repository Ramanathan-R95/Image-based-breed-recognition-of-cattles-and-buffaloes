import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
from PIL import Image, ImageOps  
import numpy as np
from tensorflow.keras.models import load_model
import os

# ------------------ Load Model ------------------

model_side =  load_model(r"Model\model_side\keras_model.h5", compile=False)

# List of breeds (ensure same order as training)
breed_list_side =open(r"Model\model_side\labels.txt", "r").readlines()
size = (224, 224)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

model_face = load_model(r"Model\model_face\keras_model.h5", compile=False)

# ------------------ Get class labels ------------------
# Assumes each subfolder name in test_dir is a class label
class_labels = open(r"Model\model_face\labels.txt", "r").readlines()
for i in range(len(class_labels)) :
    class_labels[i] = class_labels[i][:-1]
# ------------------ Preprocessing Function ------------------
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    return data

# ------------------ GUI Functions ------------------
def open_image():
    global img_path
    img_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if img_path:
        load_and_display(img_path)
def open_image_face():
    global img_path_face
    img_path_face = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if img_path:
        load_and_display2(img_path_face)
        print(img_path_face)
def load_and_display2(path):
    img = Image.open(path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    img_label2.config(image=img_tk)
    img_label2.image = img_tk
    result_label.config(text="")
def load_and_display(path):
    img = Image.open(path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk
    result_label.config(text="")  # Clear previous prediction

def predict_image():
    if not img_path:
        result_label.config(text="⚠ Please select an image first")
        return

    processed_img = preprocess_image(img_path)
    processed_img_face = preprocess_image(img_path_face)
    if processed_img is None:
        result_label.config(text="⚠ Unable to process image")
        return

    pred = model_side.predict(processed_img)
    pred_idx = np.argmax(pred, axis=1)[0]
    predicted_breed = breed_list_side[pred_idx]
    confidence = pred[0][pred_idx] * 100
    pred_face = model_face.predict(processed_img_face)
    pred_idx1 = np.argmax(pred_face, axis=1)[0]
    predicted_breed1 = class_labels[pred_idx1]
    confidence1 = pred_face[0][pred_idx1] * 100

    result_label.config(
        text=f"Predicted Breed: {predicted_breed}\nConfidence: {confidence:.2f}%"
    )
    result_label2.config(
        text=f"Predicted Breed: {predicted_breed1}\nConfidence: {confidence1:.2f}%"
    )
    final_label.config(
        text=f"Final Prediction : \n{predicted_breed1}\nConfidence: {(confidence1 + confidence)/2:.2f}%"
    )

# ------------------ Tkinter GUI ------------------
root = tk.Tk()
root.title("Cow Breed Predictor")
root.geometry("600x600")
root.config(bg = "#800020")
label = Label(root, text="Breed recognition for Cattle and Buffaloes", font=("Times", 29, "bold"),bg = "#800020",fg="white")
label.place(x=160,y= 20)

img_label = Label(root)
img_label.place(y=90,x=10)
img_label2 = Label(root)
img_label2.place(y=90,x=560)

open_btn = Button(root, text="Select Image (side)", command=open_image,font=("Times", 21, "bold"), bg="white", fg="#800020")
open_btn2 = Button(root, text="Select Image (face)", command=open_image_face,font=("Times", 21, "bold"), bg="white", fg="#800020")
open_btn.place(y=390,x=20)
open_btn2.place(x=560,y=390)

predict_btn = Button(root, text="Predict Breed", command=predict_image,font=("Times", 21, "bold"), bg="white", fg="#800020")
predict_btn.place(y=630,x=200)

result_label = Label(root, text="", font=("Times", 21, "bold"),bg = "#800020",fg="white")
result_label2 = Label(root, text="", font=("Times", 21, "bold"),bg = "#800020",fg="white")
result_label.place(x=20,y=490)
result_label2.place(x=500,y=490)
final_label =Label(root, text="", font=("Times", 21, "bold"),bg = "#800020",fg="white")
final_label.place(x=860,y=300)

img_path = ""
img_path_face = ""
root.mainloop()