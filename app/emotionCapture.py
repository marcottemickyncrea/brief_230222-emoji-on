import cv2
import numpy as np
from keras.models import load_model

emotion_model = load_model("emotions_augx2-m2-pb_degout.h5")
emotion_labels = ['Colere', 'Degout','Peur','Joie', 'Tristesse', 'Surprise', 'Neutre']
emotion_images = {
    'Colere': cv2.imread('static/emojis/colere.png'),
    'Degout': cv2.imread('static/emojis/degout.png'),
    'Peur': cv2.imread('static/emojis/peur.png'),
    'Joie': cv2.imread('static/emojis/joie.png'),
    'Tristesse': cv2.imread('static/emojis/tristesse.png'),    
    'Surprise': cv2.imread('static/emojis/surprise.png'),
    'Neutre': cv2.imread('static/emojis/neutre.png')    }

def emotionCapture():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        # Lit une frame du flux webcam
        ret, frame = cap.read()
        
        grey_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(grey_image)    
        
        #Boucle à travers les visages détectés
        for (x, y, w, h) in faces:
            # Extrait le visage
            face_cropped = grey_image[y:y+h, x:x+w]
            print("face",face_cropped.shape)
            # Redimensionner le visage pour correspondre à l'entrée du modèle
            face_resized = cv2.resize(face_cropped,(48,48),interpolation = cv2.INTER_AREA)
            face_resized = face_resized.reshape(-1,48,48,1)/255.0  
            # Prédiction de l'émotion
            emotion = emotion_labels[np.argmax(emotion_model.predict(face_resized))]
            # Affiche le smiley qui correspond ainsi que l'émotion
            emotion_image = emotion_images[emotion]            
            emotion_image = cv2.resize(emotion_image, (w, h),
                            interpolation=cv2.INTER_CUBIC)
            frame[y:y+h, x:x+w] = emotion_image
            cv2.putText(frame, emotion, (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)                

        # Renvoie la frame correspondante
        cv2.imshow('Video', frame)

        # Touche 'q' pour quitter
        key = cv2.waitKey(1)
        print(key)
        if key == ord('q'):
            break 
        
    cap.release()
    cv2.destroyAllWindows()

emotionCapture()