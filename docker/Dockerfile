FROM python:3.9
WORKDIR /mlops
COPY ./apis/app.py .
COPY ./requirements.txt .
# COPY ./clf_gamma_0.001_C_0.5.joblib .
RUN pip3 install -U scikit-learn
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN --mount=target={PWD},type=bind,source= .
EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]

# cmd to build: docker build -t exp:v1 -f docker/Dockerfile .
# cmd to run: docker run -p 5000:5000 -it exp:v1
# docker run --name mlops -p 5000:5000 -v ${pwd}/clf_gamma_0.001_C_0.5.joblib:/clf_gamma_0.001_C_0.5.joblib -it exp:v1