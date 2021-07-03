from tkinter import *
from tkinter import filedialog
import tkinter.font as font
from PIL import ImageTk,Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")


def face_and_mask(frame, faceNet, maskNet):
   (h, w) = frame.shape[:2]
   blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

   faceNet.setInput(blob)
   detections = faceNet.forward()

   faces = []
   locs = []
   preds = []

   for i in range(0, detections.shape[2]):
      prob = detections[0, 0, i, 2]
      if prob > 0.5:
         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
         (startX, startY, endX, endY) = box.astype("int")
         (startX, startY) = (max(0, startX), max(0, startY))
         (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
         face = frame[startY:endY, startX:endX]
         face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
         face = cv2.resize(face, (224, 224))
         face = img_to_array(face)
         face = preprocess_input(face)
         faces.append(face)
         locs.append((startX, startY, endX, endY))

   if len(faces) > 0:
      faces = np.array(faces, dtype="float32")
      preds = maskNet.predict(faces, batch_size=32)

   return (locs, preds)


def mark_face(frame, locs, preds):
   for (box, pred) in zip(locs, preds):
      (startX, startY, endX, endY) = box
      (mask, withoutMask) = pred

      label=''
      color=()
      percentage=0

      if mask>0.80:
         label='Mask'
         color = (0, 255, 0)
         percentage = mask*100
      else:
         label='No Mask'
         color = (0, 0, 255)
         percentage = withoutMask*100

      label = "{}: {:.2f}%".format(label, percentage)

      cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
      cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
   return frame


root = Tk()
root.title('Face Mask Detector')
root.resizable(width=False, height=False)
root.iconbitmap('images/mask.ico')

def image():
   root.filename = filedialog.askopenfilename(initialdir='', title='Select Image', filetypes=(("all files","*.*"),("png files","*.png"),("jpg files","*.jpg"),("jpeg files","*.jpeg")))
   img_path = root.filename     

   try:
      frame = cv2.imread(img_path)
      frame = imutils.resize(frame, width=400)
   except:
      return

   (locs, preds) = face_and_mask(frame, faceNet, maskNet)
   frame = mark_face(frame, locs, preds)

   cv2.imshow("Face Mask Detector", frame)


def video():
   vs = VideoStream(src=0).start()
   # loop over the frames from the video stream
   while True:
      frame = vs.read()
      frame = imutils.resize(frame, width=400)

      (locs, preds) = face_and_mask(frame, faceNet, maskNet)
      frame = mark_face(frame, locs, preds)

      cv2.imshow("Face Mask Detector", frame)

      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):
         break

   cv2.destroyAllWindows()
   vs.stop()


cover_img = ImageTk.PhotoImage(Image.open('images/cover.png').resize((950, 550), Image.ANTIALIAS))

my_label = Label(image=cover_img)
my_label.grid(row=0,column=0,columnspan=6)

buttonFont = font.Font(family='Helvetica', size=16, weight='bold')

btn1 = Button(root, text="Image Upload", command=image, padx=40, pady=20, bg='#407bff', fg='#ededed', font=buttonFont)
btn2 = Button(root, text=" Live Video ", command=video, padx=40, pady=20, bg='#407bff', fg='#ededed', font=buttonFont)

btn1.grid(row=0, column=2, pady=30)
btn2.grid(row=0, column=3, pady=30)

root.mainloop()