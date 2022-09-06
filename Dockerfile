FROM python:3.10.6-buster
ADD . /app
WORKDIR /app
#ENV
#COPY /home/lepton/code/vivekptl9/Anaemia_Classification/py /usr/nginx/Anaemia_Classification
#CMD [ "nginx", "-g", "daemon off;" ]   
CMD [ "python", "CNN-RF.py" ]