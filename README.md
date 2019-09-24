# Human And Enemy Detector
Bu projede eğitilen model insanı "human", quadcopter'i ise "enemy" etiketiyle tanımaktadır. Tensorflow-Object-Detection-API kullanılmıştır. COCO modellerinden hız ve isabet oranı dikkate alınarak faster_rcnn_inception_v2_coco modeli üzerinde eğitim gerçekleştirilmiştir. Asıl model 460 fotoğraf kullanılarak 156800 adım eğitilmiştir. Eğitilen asıl model üzerinden gerçekleştirilen test çıktısı aşağıdaki gibidir.


                  ![output](https://github.com/busracinar/HumanAndEnemyDetector/blob/master/doc/output.PNG)
