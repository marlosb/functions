import datetime
import os

from azure.storage.blob import ContainerClient
import pandas as pd
import numpy as np
from sklearn import linear_model

start_time = datetime.datetime.now()

key = os.environ.get('STORAGE_ACCOUNT_KEY')
container_url = 'https://asnrocks.blob.core.windows.net/auto'

model_path = 'models/'

def get_model(model_path, container_url, key):
    ''' List all files on model_path, order by name,
        download the last one (assuming timestamp on name),
        import the model and return its object and name
    '''
    # instanciate container client
    container_client = ContainerClient.from_container_url(
                                        container_url=container_url,
                                        credential=key)
    # list all files on model_path
    models_list = []
    for model in container_client.list_blobs(
                                         name_starts_with = model_path):
        models_list.append(model['name'])
    
    # sort all models and get last one
    models_list.sort()
    last_model_name = models_list[-1]
    # instanciate blob client
    blob_client = container_client.get_blob_client(
                                                 blob = last_model_name)
    # download model to local disk
    with open(last_model_name, 'wb') as my_blob:
        blob_data = blob_client.download_blob()
        blob_data.readinto(my_blob)
    # close handles
    blob_client.close()
    container_client.close()
    # read model from disk
    model = pd.read_pickle(last_model_name)
    
    return model, last_model_name

def get_score(params_dict, model_package):
    ''' Receive all features in a a dict
        Predicts score and returns it
    '''
    model = model_package.model
    columns = model_package.fit_vars
    
    # checks if calculated features are present, if yes create it
    if 'cylinder_displacement' in columns:
        params_dict['cylinder_displacement'] = (
                 params_dict['cylinders'] * params_dict['displacement'])
        
    if 'specific_torque' in columns:
        params_dict['specific_torque'] = (params_dict['horsepower'] 
                                 * params_dict['cylinder_displacement'])
        
    if 'fake_torque' in columns:
        params_dict['fake_torque'] = (params_dict['weight'] 
                                      / params_dict['acceleration'])
        
    # predict
    score = model.predict(pd.DataFrame.from_dict({1: params_dict}, 
                                                 orient='index'))
    return score

if __name__ == '__main__':
    # Create a test case
    input_dict = {'cylinders': 8,
                'displacement': 320,
                'horsepower': 150,
                'weight': 3449,
                'acceleration': 11.0,
                'year': 70,
                'origin': 1}

    model_package, model_version = get_model(model_path, container_url, 
                                            key)

    score = get_score(input_dict, model_package)

    elapsed_time = datetime.datetime.now() - start_time

    results_dict = {
                'Start time': start_time.strftime('%Y-%m-%d--%H-%M-%S'),
                'model version': model_version,
                'input data': input_dict,
                'predicted score': score[0],
                'scoring time': elapsed_time.total_seconds()}
    
    print(results_dict)