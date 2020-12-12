#Import modules
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

#Importing common libraries
import numpy as np
import cv2 
import random
import matplotlib.pyplot as plt
from PIL import Image


#Import width control
from width_control import *

#Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

#Importing FDK modules
from src.core.detect import Detector
from src.core.utils import utils 

#More Modules
from google.colab.patches import cv2_imshow
from scipy.spatial import distance
import pandas as pd
import os
import re
from os.path import isfile, join

st.set_page_config(
    page_title="Socio-Companion",
    page_icon=":handshake:",
    layout="centered",

    initial_sidebar_state="collapsed",
)

select_block_container_style()

st.markdown("<h1 class = 'title_name'>Socio-Companion</h1>", unsafe_allow_html=True)
st.markdown("<hr class='line'>", unsafe_allow_html=True)
st.markdown("<h2 class = 'sub_header'>Project Description</h2>", unsafe_allow_html=True)
st.markdown("\n\n")
st.markdown("<h3 class = 'small_header'>Problem Statement</h3>", unsafe_allow_html=True)
st.markdown("""<p class="problem-statement">Despite of the government's initiative to lower down the spread of Covid-19 through Movement Control Order(MCO), there's still numerous reports of people breaching the
    social distancing requirements and getting fined. Enforcement parties like police officers are risking their lives to curb and reduce the infection in community. In this case, robots are recommended to perform
    patrol and give our warnings to the public. With the application of advance image recognition technology, robots are able
    recognize and classify objects. However,  effective and proper mechanism of robotics is important to facilitate enforcement officers. The use of robots in performing screening will reduce
    the infection risk of enforcement officers and the society should support and care for the developments
    in the robotics technology as this will be beneficial for the people especially healthcare sector which has already burdened by current pandemic. Many tasks which are beyond the human
    ability can be performed with the help of robotics. The usage of social robots are applied to selected
    tasks and used in areas where humans are not capable of performing or more riskier to complete.</p>""", unsafe_allow_html=True)
st.markdown("<h3 class = 'small_header'>Project Technical Description</h3>", unsafe_allow_html=True)
st.markdown("""<p class="problem-statement">This project contains Detectron2 to  make the model more accurate in
    determining the distance of people. Detectron2 is used in detecting people and detecting the distance between people who are walking near.</p>""", unsafe_allow_html=True)

if not os.path.isdir("/content/FDK/temp_images"):
    os.mkdir("/content/FDK/temp_images")

if not os.path.isdir("/content/FDK/final_images"):
    os.mkdir("/content/FDK/final_images")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h1 class = 'title_name'>Socio-Companion UI</h1>", unsafe_allow_html=True)
#Importing photos
st.markdown("<h3 class = 'small_header'>Video Uploading Section.</h3>", unsafe_allow_html=True)

file_path = st.text_input(label="Specify file path and name for the video that you would like to upload.")
#Video Capturing
vidcap = cv2.VideoCapture('{path}'.format(path = file_path))
#File function to
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("/content/FDK/temp_images/image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
    
sec = 0
frameRate = 0.5 #it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

#Downloading the pretrained module from Detectron2's model zoo that
#is ready for prediction

cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9 

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
predictor = DefaultPredictor(cfg)

#img = Image.open("temp_images/image1.jpg")
#img = utils.pil_to_cv2(img)
#outputs = predictor(img)
#
#v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#cv2_imshow(v.get_image()[:, :, ::-1])
#
#classes=outputs["instances"].pred_classes.cpu().numpy()
#print(classes)
#
#bbox = outputs["instances"].pred_boxes.tensor.cpu().numpy()
#print(bbox)
#
#ind = np.where(classes==0)[0]
#
#person=bbox[ind]
#
#num=len(person)
#
#x1,y1,x2,y2 = person[0]
#print(x1,y1,x2,y2)
#
#img = cv2.imread("temp_images/image1.jpg")
#_ = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
#
#plt.figure(figsize=(20,10))
#plt.imshow(img)
#
#x_center = int((x1+x2)/2)
#y_center = int(y2)
#
#center = (x_center, y_center)
#
#_ = cv2.circle(img, center, 5, (255, 0, 0), -1)
#plt.figure(figsize=(20,10))
#plt.imshow(img)

def mid_point(img,person,idx):
    x1,y1,x2,y2 = person[idx]
    _ = cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 0), 2)

    x_mid = int((x1+x2)/2)
    y_mid = int(y2)
    mid = (x_mid, y_mid)

    _ = cv2.circle(img, mid, 5, (255, 0, 0), -1)
    cv2.putText(img, str(idx), mid, cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2, cv2.LINE_AA)

    return mid

#midpoints = [mid_point(img,person,i) for i in range(len(person))]

#plt.figure(figsize=(20,10))
#plt.imshow(img)

def compute_distance(midpoints, num):
    dist = np.zeros((num,num))
    for i in range(num):
        for j in range(i+1, num):
            if i!=j:
                dst = distance.euclidean(midpoints[i], midpoints[j])
                dist[i][j] = dst
    return dist

#dist = compute_distance(midpoints,num)

def find_closest(dist,num,thresh):
    p1=[]
    p2=[]
    d=[]
    for i in range(num):
        for j in range(i, num):
            if((i!=j)& (dist[i][j] <= thresh)):
                p1.append(i)
                p2.append(j)
                d.append(dist[i][j])

    return p1,p2,d

#thresh = 100
#p1,p2,d=find_closest(dist,num,thresh)
#df = pd.DataFrame({"p1": p1, "p2": p2, "dist":d})
#print(df)

def change_2_red(img, person,p1,p2):
    risky = np.unique(p1+p2)
    for i in risky:
        x1,y1,x2,y2 = person[i]
        _ = cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
        return img

#img = change_2_red(img, person, p1,p2)

#plt.figure(figsize=(20,10))
#plt.imshow(img)

def find_closest_people(value,thresh):

  img = cv2.imread('/content/FDK/temp_images/image' + str(value) + '.jpg')
  outputs = predictor(img)
  classes=outputs['instances'].pred_classes.cpu().numpy()
  bbox=outputs['instances'].pred_boxes.tensor.cpu().numpy()
  ind = np.where(classes==0)[0]
  person=bbox[ind]
  midpoints = [mid_point(img,person,i) for i in range(len(person))]
  num = len(midpoints)
  dist= compute_distance(midpoints,num)
  p1,p2,d=find_closest(dist,num,thresh)
  img = change_2_red(img,person,p1,p2)
  cv2.imwrite('/content/FDK/final_images/new_image' + str(value) + '.jpg',img)
  return 0

predict = False
def getPhoto(var):
    st.image("/content/FDK/final_images/final_image" + str(var) + ".jpg")

for a in range(1, count - 1, 1):
    thresh = 100
    find_closest_people(a, thresh)
    predict = True

if predict == True:
    values = st.slider(label="Changing Photos", min_value = 1, max_value = count - 1, step = 1)
    getPhoto(values)
    if st.button("Save Video"):
        pathIn= '/content/FDK/final_images'
        pathOut = 'final_vid.mp4'
        fps = 0.5
        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        #for sorting the file names properly
        files.sort(key = lambda x: x[5:-4])
        files.sort()
        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        #for sorting the file names properly
        files.sort(key = lambda x: x[5:-4])
        for i in range(len(files)):
            filename=pathIn + files[i]
            #reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            
            #inserting the frames into an image array
            frame_array.append(img)
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()
