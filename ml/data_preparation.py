import pandas as pd
from sklearn.utils import resample

def sample_class(train_majority,train_minority,count_new_class,action,random_state=123):
    if action == 'up_sample_minority_class':
        train_minority_upsampled = resample(train_minority
                                           ,replace=True
                                           ,n_samples=count_new_class
                                           ,random_state=random_state)
        train_upsampled = pd.concat([train_majority,train_minority_upsampled])
        balanced_data = train_upsampled.copy()
    elif action == 'down_sample_majority_class':
        train_majority_downsampled = resample(train_majority
                                             ,replace = False
                                             ,n_samples=count_new_class
                                             ,random_state=random_state)
        train_downsampled = pd.concat([train_majority_downsampled,train_minority])
        balanced_data = train_downsampled.copy()
    else:
        return None
    return balanced_data

