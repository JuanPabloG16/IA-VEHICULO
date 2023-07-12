#importamos librerias
import torch
import cv2
import numpy as np

#importamos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/gutie/Downloads/carros/model/car.pt')

#Realizamos videocaptura
cap = cv2.VideoCapture(1)

# Empezamos
while True:
    #Realizar lectura de la video captura
    ret, frame = cap.read()
    #Realizamos detección
    results = model(frame)

    #mostramos fps
    cv2.imshow('Detector de carros', np.squeeze(results.render()))

    #Leer el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()



