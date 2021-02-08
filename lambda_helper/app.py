print('container start')
try:
  import unzip_requirements
except ImportError:
  pass
print('unzip end')

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import boto3
import os
import tempfile
# import requests

#Pull up the env variables
MODEL_BUCKET_NAME = os.environ['MODEL_BUCKET_NAME']
MODEL_FILE_NAME_KEY = os.environ['MODEL_FILE_NAME_KEY']
TEMP_DIR = '/tmp' 
MODEL_PATH = os.path.join(TEMP_DIR, MODEL_FILE_NAME_KEY)

#Download model from S3
print('downloading model...')
s3 = boto3.resource('s3')
s3.Bucket(MODEL_BUCKET_NAME).download_file(MODEL_FILE_NAME_KEY, MODEL_PATH)

print('loading model...')
#Load the model for Predictions
model = load(MODEL_PATH)
print('model loaded\n')


def lambda_handler(event, context):
    body = dict()
    
    if (event['httpMethod'] == 'GET'):
        params = event['queryStringParameters']
    else:
        params = json.loads(event['body'])
    
    if params is not None and 'words' in params:
        x = [params['words']]
        print(x)
        
        # predict document classes and decode predictions
        predictions = model.predict(x)
        confidence_mat = model.predict_proba(x)
        
        body['prediction'] = predictions[0]
        body['confidence'] = confidence_mat.max()
    
    #Create the response
    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Content-Type": "application/json",
            "X-Custom-Header": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, PATCH, DELETE",
            "Access-Control-Allow-Headers": "X-Requested-With,content-type"
        }

    }

    return response