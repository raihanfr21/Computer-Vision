import cv2
import numpy as np

cap = cv2.VideoCapture('assets\SIMPANG BANDARA SSQ.mp4')

min_width_react = 80
min_height_react = 80 

count_line_position = 550

algo = cv2.createBackgroundSubtractorMOG2()

def center_handle(x,y,w,h):
  x1 = int(w/2)
  y1 = int(h/2)
  cx = x+x1
  cy = y+y1
  return cx,cy

detect = []
offset=6 
counter = 0

while True:
  ret,frame1= cap.read()
  grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(grey, (3,3),5)
  
  img_sub = algo.apply(blur)
  dilat = cv2.dilate(img_sub,np.ones((5,5)))
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
  dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
  dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
  counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
  cv2.line(frame1,(25,count_line_position),(12000,count_line_position),(255,127,0),3)

  
  for (i,c) in enumerate(counterShape):
    (x,y,w,h) = cv2.boundingRect(c)
    validate_counter = (w>= min_width_react) and (w>= min_height_react)
    if not validate_counter:
      continue

    cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
  
    center = center_handle(x,y,w,h)
    detect.append(center)
    cv2.circle(frame1,center,4, (0,0,255),-1)

    for (x,y) in detect:
      if y<(count_line_position+offset) and y>(count_line_position-offset):
        counter+=1
      cv2.line(frame1,(25,count_line_position),(12000,count_line_position),(255,127,0),3)
      detect.remove((x,y))
      print("Jumlah Kendaraan:"+str(counter))

  text_color = (255, 255, 255)
  background_color = (0, 0, 0)

  text_location = (frame1.shape[1] - 10, 30)
  text_size = cv2.getTextSize("JUMLAH KENDARAAN:" + str(counter), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

  background_size = (text_size[0] + 10, text_size[1] + 10)
  background_location = (text_location[0] - text_size[0] - 5, text_location[1] - 5)

  cv2.rectangle(frame1, background_location, (background_location[0] + background_size[0], background_location[1] + background_size[1]), background_color, -1)
  cv2.putText(frame1, "Jumlah Kendaraan:" + str(counter), (background_location[0] + 5, background_location[1] + text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
  text_color, 2, cv2.LINE_AA, False)
  
  # cv2.imshow('Detector',dilatada)
  cv2.imshow('Deteksi Kendaraan', frame1)

  if cv2.waitKey(20) == 13:
    break
  
cv2.destroyAllWindows()
cap.release()