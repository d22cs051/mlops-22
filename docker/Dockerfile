FROM python:3.9
WORKDIR /mlops
COPY ./apis/app.py .
COPY ./requirements.txt .
COPY ./clf_gamma_0.001_C_0.5.joblib .
RUN pip3 install -U scikit-learn
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
