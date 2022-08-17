import tflite_runtime.interpreter as tflite
import tensorflow as tf
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model dosya yolu',
                    default='C://Users//acer//Desktop//YOUTUBE2//models-master//research//object_detection//cemberssdmodel//saved_model//cember_modeli.tflite')
parser.add_argument('--labels', help='etiketlerin dosya yolu',
                    default='C://Users//acer//Desktop//YOUTUBE2//models-master//research//object_detection//cemberssdmodel//saved_model//label.txt')
# parser.add_argument('--video', help='video dosya yolu',
#                     default='test.mp4')
parser.add_argument('--threshold', help='minimum doğruluk oranı',
                    default=0.5)
                    
args = parser.parse_args()

# model yolu değişkene atanıyor
PATH_TO_MODEL_DIR = args.model

# etiket yolu değişkene atanıyor
PATH_TO_LABELS = args.labels

# video yolu değişkene atanıyor
#VIDEO_PATH = args.video

# minimum doğruluk oranı float tipinde değişkene atanıyor
MIN_CONF_THRESH = float(args.threshold)

import time
print('Model yükleniyor...', end='')
start_time = time.time()

# TFLite model yükleniyor
interpreter = tf.contrib.lite.Interpreter(model_path=PATH_TO_MODEL_DIR)
# Etiketler yükleniyor
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
end_time = time.time()
elapsed_time = end_time - start_time
print('Tamamlandı! İşlemler {} saniye sürdü'.format(elapsed_time))

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Saniyede görütülenen kare hız işlemi başlatılıyor
frame_rate_calc = 1
freq = cv2.getTickFrequency()
# print('Model çalışması {}  saniye sürdü... '.format(VIDEO_PATH), end='')
# Video başlatılıyor
video = cv2.VideoCapture(1)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
counter=1
fileobj = open('test1.txt', 'a')

while(video.isOpened()):
    # FPS(Saniyede işlenen görüntü sayısı için zaman başlatıldı.
    
    t1 = cv2.getTickCount()
    
    # Videodan bir kare alınıyor
    ret, frame1 = video.read()

    # Resmi veri girişi için gerekli formata çeviriyoruz.
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Tespit çalışması için gelen resmi işleme alıyoruz.
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Tespit sonuçlarını alıyoruz. 
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Tespit kutusu 
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Tespit sınıfı 
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Doğrulu oranı
   

    # Eğer doğruluk oranı istenilenin üstünde ise tüm algılanan nesneler için tespit kutusunu çiziyoruz.
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):

            # Tespit kutusunun koordinatları alınıp çizim yapılıyor
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Etiketi yazdırıyoruz.
            object_name = labels[int(classes[i])] # etiketi etiket dosyamızdan çekiyoruz.
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Örnek: 'eagle: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Font boyutu ayarlanıyor
            label_ymin = max(ymin, labelSize[1] + 10) # label konumunu düzenliyoruz
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # etiket için beyaz bir kutu çiziyoruz
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # etiketi yazdırıyoruz
            
            # fileobj.write("Id:{} Kuşun İsmi:{} Doğruluk Oranı:{} \n".format(counter,object_name,int(scores[i]*100)) 
            
            # uzanti='jpg'
            # isim='{0}{1}.{2}'.format(object_name,counter,uzanti)
            # print(isim)
            # cv2.imwrite(isim, frame)
            # counter=counter+1
            
            
    # FPS i yazdırıyoruz
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(15,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    # FPS, etiket ve tespit kutusu ile birlikte resmi bastırıyoruz.
    cv2.imshow('Object Detector', frame)
            

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
print("Done")
