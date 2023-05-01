FROM python:3.9
EXPOSE 8080
WORKDIR /app
RUN apt update -y
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD python app.py 
