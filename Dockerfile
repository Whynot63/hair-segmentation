FROM tensorflow/tensorflow:latest

RUN apt -y install unzip wget apt libsm6 libxrender1 libxext6
RUN pip3 install numpy gdown tensorflow keras scikit-image opencv-python

COPY src /hair-recognition
WORKDIR /hair-recognition