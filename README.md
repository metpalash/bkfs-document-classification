# BKFS Document Classification

### Problem Description - 
This dataset represents the output of the OCR stage of our data pipeline. 
We need to train a document classification model. Deploy the model to a public cloud platform (AWS/Google/Azure/Heroku) as a webservice with a simple ui.

### Project Folder Structure -
  - Model Specific Files: ```ml/```

  - Restful Api: ```lambda_helper/```

  - FrontEnd Page: ```front_end/```

The application uses several AWS resources, including Lambda functions and an API Gateway API. These resources are defined in the `template.yaml` file in this project.

#### Prediction UI
```
https://bkfsmodel.s3.amazonaws.com/index.html
```

#### Rest API
```
https://et9cl4lp4l.execute-api.us-east-1.amazonaws.com/Prod/predict/
```

#### Sample Curl Request
- GET - 
```
curl --request GET 'https://et9cl4lp4l.execute-api.us-east-1.amazonaws.com/Prod/predict?words=putDocumentTextHere'
```
- POST - 
```
curl --location --request POST 'https://et9cl4lp4l.execute-api.us-east-1.amazonaws.com/Prod/predict' \
--header 'Content-Type: application/json' \
--data-raw '{"words":"putDocumentTextHere"}'
```

### Code Deploy to AWS -
This project contains source code and supporting files for a serverless application that you can deploy with the SAM CLI.
To deploy the code to AWS

The project is created with: ```Python 3.6```

libraries: ```Scikit-learn, Pandas, Numpy, Seaborn, matplotlib, joblib, boto3```

You can use ```requirements.txt``` to create a venv

 - ##### Clone the Git Repo -
```bash
git clone https://github.com/metpalash/bkfs-document-classification.git
```

 - ##### Stage data - 
Put the data in  ```ml/data``` directory as 'shuffled-full-set-hashed.csv'

 - ##### Train Model
Navigate to ml folder and run, this will train the model and export the model as '.joblib'
```bash
python train.py
```
 - ##### Deploy Model
Deploy the model to S3 bucket.

 - ##### Env Variables
Update the enviornment variables in template.yaml file with the ones you have - 
```
MODEL_BUCKET_NAME: bkfsmodel
MODEL_FILE_NAME_KEY: mlSGDClassifier.joblib

```

 - ##### AWS SAM Installation
Download the SAM CLI & Docker

* SAM CLI - [Install the SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
* Docker - [Install Docker community edition](https://hub.docker.com/search/?type=edition&offering=community)

- ##### Create an AWS ECR Repository - 
```bash
aws ecr create-repository --repository-name bkfs-doc-class-repo --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true
```
- ##### Build and Deploy AWS lambda fn and Resrouces
Run the following Command - enter the ecr repository from previous step during
deploy process whenever asked -
```bash
sam build
sam deploy --guided
```

- ##### UI Deploy
UI can be run both locally as well as you can deploy to S3 bucket as static website.
Please update the API endpoint from the previous step to the index.html 
```
var url1 = "https://et9cl4lp4l.execute-api.us-east-1.amazonaws.com/Prod/predict/";
```

### Summary -

- ```ml/data_exploration.ipynb```:

I started with data analysis and data pre-processing from our dataset. 

- ```ml/model.ipynb``` :

Then I have used CountVectorizer and TF-IDF to convert the data into vectors. I have also experimented with several Machine Learning algorithms: Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, KNeighbour Classifier, Stochastic Gradient Descent and MLP. For the modeling i have utilized sklearn pipeline for all the modeling steps.
I also tried to include SelectKBest feature using chi2 to extract relevant features from the sparse data, but it didnt help
much in improving the overall accuracy.
After getting the best pick among the algorithms, i have performed grid search to perform the hyperparameter tuning.

- ```ml/train.py```:

This is a python file you can run to train the best model identified in previous step.
It will train from the raw csv and export the model as '*.joblib'.

From our experiments we can see that the tested models give a overall high accuracy . The SVM (Count Vector +TF-IDF) model and SGD Classifier(Count Vector +TF-IDF) model gives the best accuracy of validation set.

| Model              | Embeddings    | Accuracy |
| ------------------ |:-------------:| --------:|
| Naive Bayes        | CV+TF-IDF     | 0.73     |
| Random Forest      | CV+TF-IDF     | 0.85     |
| SGD                | CV+TF-IDF     | 0.88     |
| Logistic Regression| CV+TF-IDF     | 0.86     |
| LinearSVM          | CV+TF-IDF     | 0.88     |
| KNeighbour         | CV+TF-IDF     | 0.82     |


##### Best Performers-
SGD and LinearSVM

##### Classification Report-

```
SGDClassifier accuracy : 0.875876492121598
                         
                         precision    recall  f1-score   support

            APPLICATION       0.68      0.83      0.75        47
                   BILL       0.92      0.85      0.88      2544
            BILL BINDER       0.28      0.46      0.35        71
                 BINDER       0.87      0.89      0.88      2054
    CANCELLATION NOTICE       0.87      0.90      0.88      2358
     CHANGE ENDORSEMENT       0.87      0.90      0.89       181
            DECLARATION       0.51      0.33      0.40       227
            
   DELETION OF INTEREST       0.92      0.91      0.92      1154
      EXPIRATION NOTICE       0.80      0.89      0.84       142
INTENT TO CANCEL NOTICE       0.65      0.64      0.64        61
     NON-RENEWAL NOTICE       0.85      0.93      0.89       150
          POLICY CHANGE       0.85      0.86      0.85      2399
   REINSTATEMENT NOTICE       0.93      0.97      0.95      1010
         RETURNED CHECK       0.91      0.92      0.91       168

               accuracy                           0.88     12566
              macro avg       0.78      0.80      0.79     12566
           weighted avg       0.87      0.87      0.88     12566
```