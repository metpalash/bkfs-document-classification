# BKFS Document Classification

###### Problem Description - 
This dataset represents the output of the OCR stage of our data pipeline. 
We need to train a document classification model. Deploy the model to a public cloud platform (AWS/Google/Azure/Heroku) as a webservice with a simple ui.

###### Project Folder Structure -
-Model Specific Files: ```ml/```

-Restful Api: ```lambda_helper/```

-Front end Page: ```front_end/```


###### Data Description -
The data consist of 14 different category of documents


###### Data Description -
To deploy the code to AWS

Clone the Git Repo -
```bash
git clone https://github.com/metpalash/bkfs-document-classification.git
```

Download the SAM CLI/Docker

* SAM CLI - [Install the SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
* Docker - [Install Docker community edition](https://hub.docker.com/search/?type=edition&offering=community)

Create an AWS ECR Repository - 
```bash
aws ecr create-repository --repository-name bkfs-doc-class-repo --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true
``

Run the following Command - enter the ecr repository from previous step during
deploy process whenever asked -
```bash
sam build
sam deploy --guided
```



Project contains:

lambda_helper/*
For the lambda helper function

app.py - The lambda helper function

ml/*
For all model specific files

For all the data exploration and preprocessing - 
data_exploration.ipynb

For Modeling-
model.ipynb

For training the final model-
train.py

Summary
We begin with data analysis and data pre-processing from our dataset. Then we have used a few combination of text representation such as BoW and TF-IDF and we have trained the word2vec and doc2vec models from our data. We have experimented with several Machine Learning algorithms: Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, Gradient Boosting and MLP and Convolutional Neural Network (CNN) using different combinations of text representations and embeddings.

From our experiments we can see that the tested models give a overall high accuracy and similar results for our problem. The SVM (BOW +TF-IDF) model and MLP model give the best accuracy of validation set. Logistic regression performed very well both with BOW +TF-IDF and Doc2vec and achieved similar accuracy as MLP. CNN with word embeddings also has a very comparable result (0.93) to MLP.

Model	Embeddings	Accuracy
Logistic Regression	BOW +TF-IDF	0.91
SVM	BOW +TF-IDF	0.93
Naive Bayes	BOW +TF-IDF	0.90
Random Forest	BOW +TF-IDF	0.91
Gradient Boosting	BOW +TF-IDF	0.91
Logistic Regression	Doc2vec (DBOW)	0.91
Logistic Regression	Doc2vec (DM)	0.89
SVM	Doc2vec (DBOW)	0.93
MLP	Word embedding	0.93
CNN	Word embedding	0.93
The project is created with:
Python 3.6
libraries: NLTK, Gensim, Keras, Scikit-learn, Pandas, Numpy, Seaborn, pyLDAvis.
Running the project:
To run this project use Jupyter Notebook or Google Colab.

This project contains source code and supporting files for a serverless application that you can deploy with the SAM CLI. It includes the following files and folders.

- hello_world - Code for the application's Lambda function and Project Dockerfile.
- events - Invocation events that you can use to invoke the function.
- tests - Unit tests for the application code. 
- template.yaml - A template that defines the application's AWS resources.

The application uses several AWS resources, including Lambda functions and an API Gateway API. These resources are defined in the `template.yaml` file in this project. You can update the template to add AWS resources through the same deployment process that updates your application code.


## Use the SAM CLI to build and test locally

Build your application with the `sam build` command.

```bash
bkfs-document-classification$ sam build
```

The SAM CLI builds a docker image from a Dockerfile and then installs dependencies defined in `hello_world/requirements.txt` inside the docker image. The processed template file is saved in the `.aws-sam/build` folder.

Test a single function by invoking it directly with a test event. An event is a JSON document that represents the input that the function receives from the event source. Test events are included in the `events` folder in this project.

Run functions locally and invoke them with the `sam local invoke` command.

```bash
bkfs-document-classification$ sam local invoke HelloWorldFunction --event events/event.json
```

The SAM CLI can also emulate your application's API. Use the `sam local start-api` to run the API locally on port 3000.

```bash
bkfs-document-classification$ sam local start-api
bkfs-document-classification$ curl http://localhost:3000/
```


## Unit tests

Tests are defined in the `tests` folder in this project. Use PIP to install the [pytest](https://docs.pytest.org/en/latest/) and run unit tests from your local machine.

```bash
bkfs-document-classification$ pip install pytest pytest-mock --user
bkfs-document-classification$ python -m pytest tests/ -v
```

Recommendations:
Here are some recommendations that can be explored to further improve the analysis:

I havent used sklearn's pipeline function which gives a lot of order of the steps involved in training, predicting the classifier.
Using Functional programming, we can form a general function which can be used to pass classifiers and to derive the results. I have attached a separate notebook which contains such code that I wrote.
The current version of the code can be made much more better by making it more modular and defining classes. If required, I can expedite on that as well.
I've made use of the state of the art text classification algorithms after going through series of research papers. We can also use Neural networks (MLPs) as well and if needed, it can be implemented.
Using latent factorization methods like Non-negative matrix factorization, we can find higher level features that can then be used during classification. I have done one such analysis by applying sparse coding on a transactional database to find out basis vectors/dictionary which improved classification results. The same can be done here as well.
We can further augment the feature extraction process by assigning different weights to the text in different positions e.g. assigning more weight to the text in title and the text at the starting sections of the body. This could be used to explore if it improves the results or not.
We can also explore forming ngram features to see if those generate any better results or not.
Similarly, we can use advanced methodology like word2vec to find out words that occur together and can use them in the features extraction process as well.
We can experiment with other feature selection methods like Mutual Information gain to see which one gives better results.
Another method to validate the results of classifiers can be Area under the curve (AUC) of ROC curve. Using One-Vs-All classification, we can form AUC to further assess the performance of our classifiers.