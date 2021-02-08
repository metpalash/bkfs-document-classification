#Loading all the modules
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from data_preparation import sample_class
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from joblib import dump
import os



def train(data_path='./data/shuffled-full-set-hashed.csv',model_nm = 'SGDClassifier',filepath=None):
    
    #Step-1 - Data Loading
    #data load
    print('loading data')
    df = pd.read_csv(data_path,header=None,names=['type','text'])

    #Step-2 Cleaning and Preporcessing Data
    #deal with null values
    df.dropna(inplace=True)

    #drop duplicates
    df = df.drop_duplicates()

    #Genrate a new feature which is the total len of document
    df['word_count'] = df['text'].str.split().str.len()

    #drop the docs having less than 10 words
    df = df[df['word_count'] > 10]

    #Drop the outliers
    df = df[~((df['type']=='POLICY CHANGE') & (df['word_count']>1100))]
    df = df[~((df['type']=='BILL') & (df['word_count']>1300))]
    df = df[~((df['type']=='BINDER') & (df['word_count']>1400))]
    df = df[~((df['type']=='CANCELLATION NOTICE') & (df['word_count']>750))]
    df = df[~((df['type']=='REINSTATEMENT NOTICE') & (df['word_count']>250))]
    df = df[~((df['type']=='DELETION OF INTEREST') & (df['word_count']>250))]
    df = df[~((df['type']=='DECLARATION') & (df['word_count']>1750))]
    df = df[~((df['type']=='EXPIRATION NOTICE') & (df['word_count']>600))]
    df = df[~((df['type']=='BILL BINDER') & (df['word_count']>1100))]
    df = df[~((df['type']=='CHANGE ENDORSEMENT') & (df['word_count']>200))]

    #Use the downsampling function to reduce the bill to 10000
    df= sample_class(df[df['type']=='BILL']
                    ,df[df['type']!='BILL']
                    ,10000
                    ,'down_sample_majority_class'
                    ,random_state=123)
    
    # Defining features and target columns
    X = df['text']
    y = df['type']

    #Lets split the data into train and test
    # Split the dataset into train and test sets
    print('Spliting Data into train and test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

    #Step-3 Modeling
    # Create SGDClassifier model pipeline
    if (model_nm == 'SGDClassifier'):
        
        model = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1,2))),
                        ('tfidf', TfidfTransformer()),
                        ('model',SGDClassifier(max_iter=1000,loss='modified_huber',class_weight='balanced')), ]
                        ,verbose=True)
    elif(model_nm=='LinearSVC'):
        # Create LinearSVC model pipeline
        model = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1,2))),
                          ('tfidf', TfidfTransformer()),
                          ('model',LinearSVC()), ],verbose=True)
    else:
        return ('Invalid Model Name')
    
    #Lets fit the model
    print('Training Model')
    model.fit(X_train,y_train)
    print('Training is complete')

    #Lets test the model
    # Make predictions

    ytest = np.array(y_test)
    pred_y = model.predict(X_test)

    # Evaluate predictions
    print('{} accuracy : {}'.format(model_nm,accuracy_score(pred_y, y_test)))
    print(classification_report(ytest, pred_y))
    print(confusion_matrix(ytest, pred_y,labels=y_test.unique()))

    # Save model
    if filepath is None:
       filepath = os.getcwd() + model_nm +'.joblib'
    else:
        filepath = filepath + model_nm + '.joblib'
    print('Saving Model')
    dump(model,filepath)
    print('Model Saved at :',filepath)

    return True

if __name__ == '__main__':
    train()
