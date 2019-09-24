import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import json
import requests
import math
import datetime
from requests.auth import HTTPBasicAuth,HTTPDigestAuth
from dronekit import connect,LocationGlobalRelative,VehicleMode


sys.path.append("..")
vehicle = connect('192.168.137.1:14550')
s = requests.session()
a = {
	"kadi" : "iguhuranka",
	"sifre" : "ias4899n76"
}
b = s.post('http://10.0.0.9:64559/api/giris', json=a)
print(b)
print(b.text)

c = s.get('http://10.0.0.9:64559/api/sunucusaati')
print(c)
print(c.text)

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util


MODEL_NAME = 'inference_graph'

CWD_PATH = os.getcwd()


PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')


PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')


NUM_CLASSES = 2


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


video = cv2.VideoCapture(1)
i=0
ks=0
xmin = 0
ymin = 0
xmax = 0
ymax= 0
while(True):
    dikilme = int(vehicle.attitude.pitch * (180 / math.pi))
    yatis = int(vehicle.attitude.roll * (180 / math.pi))
    yonelme = int(vehicle.attitude.yaw * (180 / math.pi))
    hiz = float(round(vehicle.groundspeed, 1))
    enlem = float(vehicle.location.global_relative_frame.lat)
    boylam = float(vehicle.location.global_relative_frame.lon)
    irtifa = float(vehicle.location.global_relative_frame.alt)
    batarya = int(((25.2-vehicle.battery.voltage)*100)/6.2)
    hedef_merkez_x = int(xmin + ((xmax-xmin)/2))
    hedef_merkez_y = int(ymin + ((ymax-ymin)/2))
    hedef_genislik = int(xmax - xmin)
    hedef_yukseklik = int(ymax - ymin)
    if vehicle.mode.name == "GUIDED" : iha_otonom = 1
    else : iha_otonom = 0
    iha_otonom=1
    i=i+1
    an = datetime.datetime.now()
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    cv2.rectangle(frame, (160, 48), (480 ,432), (0,0,255), 3)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.85)
    
    box = np.squeeze(boxes)
    for index,value in enumerate(classes[0]):
        if i==1 :
            i=0
            dikilme = int(vehicle.attitude.pitch * (180 / math.pi))
            yatis = int(vehicle.attitude.roll * (180 / math.pi))
            yonelme = int(vehicle.attitude.yaw * (180 / math.pi))
            hiz = float(round(vehicle.groundspeed, 1))
            enlem = float(vehicle.location.global_relative_frame.lat)
            boylam = float(vehicle.location.global_relative_frame.lon)
            irtifa = float(vehicle.location.global_relative_frame.alt)
            batarya = int(((25.2-vehicle.battery.voltage)*100)/6.2)
            hedef_merkez_x = int(xmin + ((xmax-xmin)/2))
            hedef_merkez_y = int(ymin + ((ymax-ymin)/2))
            hedef_genislik = int(xmax - xmin)
            hedef_yukseklik = int(ymax - ymin)
            if vehicle.mode.name == "GUIDED" : iha_otonom = 1
            else : iha_otonom = 0
            x={
                "takim_numarasi": 28,
                "IHA_enlem" : float(enlem) ,
                "IHA_boylam": float(boylam) ,
                "IHA_irtifa": float(irtifa) ,
                "IHA_dikilme": int(dikilme) ,
                "IHA_yonelme": int(yonelme ),
                "IHA_yatis": int(yatis) ,
                "IHA_hiz": int(hiz) ,
                "IHA_batarya": int(100-batarya) ,
                "IHA_otonom": int(iha_otonom), 
                "IHA_kilitlenme": 1, 
                "Hedef_merkez_X": int(hedef_merkez_x) ,
                "Hedef_merkez_Y": int(hedef_merkez_y ),
                "Hedef_genislik": int(hedef_genislik ),
                "Hedef_yukseklik": int(hedef_yukseklik ), 
                "GPSSaati": {
                    "saat": int(an.hour),
                    "dakika": int(an.minute),
                    "saniye": int(an.second),
                    "milisaniye": int(an.microsecond/1000)} }
            r = s.post('http://10.0.0.9:64559/api/telemetri_gonder', json=x)
            print(r)
            print(int(an.second))
            
        if scores[0,index] > 0.5:
                if category_index.get(value)["name"] == "enemy" :     
                    for i in range(len(boxes)):
                        ymin = (int(box[i,0]*480))
                        xmin = (int(box[i,1]*640))
                        ymax = (int(box[i,2]*480))
                        xmax = (int(box[i,3]*640))
                        
                        
                        
                        if (xmin>160) & (xmax<480) & (ymin>48) & (ymax<432) & (ks == 0) :
                            ks = ks+1
                            print("Kilitlenme başladı..")  
                            kilitlenmeBaslangicZamani = {
                            "saat": int(an.hour),
                            "dakika": int(an.minute),
                            "saniye": int(an.second),
                            "milisaniye": int(an.microsecond/1000)}
                            print(kilitlenmeBaslangicZamani)

                        if (xmin>160) & (xmax<480) & (ymin>48) & (ymax<432) & (ks<20) :
                            ks=ks+1
                            print(ks)
                            r = s.post('http://10.0.0.9:64559/api/telemetri_gonder', json=x)
                            print("Kilitlenme süresi 10 sn den küçük")
                        else :
                            ks = 0
                             
                        
                        if (xmin>160) & (xmax<480) & (ymin>48) & (ymax<432) & (ks == 20) :
                            print("Kilitlenme bitti..")
                            kilitlenmeBitisZamani ={
                            "saat": int(an.hour),
                            "dakika": int(an.minute),
                            "saniye": int(an.second),
                            "milisaniye": int(an.microsecond/1000)}
                            print(ks)
                            y={
                                "kilitlenmeBaslangicZamani" : kilitlenmeBaslangicZamani, 
                                "kilitlenmeBitisZamani" : kilitlenmeBitisZamani,
                                "otonom_kilitlenme" : 0
                            }
                        
                            d= s.post('http://10.0.0.9:64559/api/kilitlenme_bilgisi', json=y)
                            print(d)
                            print(d.text)
                            print(y)
                            ks=0
                
               

                            
    cv2.imshow('Object detector', frame)
    

    
    
    
        
 
    if cv2.waitKey(1) == ord('q'):
        break

#if cv2.waitKey(1) == ord ('p') :
g = s.get('http://10.0.0.9:64559/api/cikis')
print(g)
print(g.text)
#break  

video.release()
cv2.destroyAllWindows()

