import datetime
import os

import pandas as pd
import numpy as np

import feature_engine.missing_data_imputers as mdi
from feature_engine import categorical_encoders as ce
from feature_engine import variable_transformers as vt

from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection

import matplotlib.pyplot as plt

from azure.storage.blob import ContainerClient

start_time = datetime.datetime.now()

key = os.environ.get('STORAGE_ACCOUNT_KEY')
container_url = 'https://asnrocks.blob.core.windows.net/auto'

data_path = 'data/'
data_file = 'auto.xlsx'

model_path = 'models/'
model_name = (
    f'auto_mode_{start_time.strftime("%Y-%m-%d--%H-%M-%S")}.pickle')

def read_input_data(data_path, data_file, container_url, key):
    ''' Downloads data_file from container_url 
        and import its data to a pandas Dataframe
        
        Input file must be a CSV,
        container_url and key must be from Azure Blob Storage
        
        Returns pandas DataFrame
    '''
    
    # instanciate containder client
    container_client = ContainerClient.from_container_url(
                                        container_url=container_url,
                                        credential=key)
    
    # create a blob client to the file
    blob_client = container_client.get_blob_client(blob = data_path 
                                                          + data_file)
    
    # download blob to local disk
    with open(data_path + data_file, 'wb') as my_file:
        blob_data = blob_client.download_blob()
        blob_data.readinto(my_file)
        
    # close handles
    blob_client.close()
    container_client.close()
    
    # read file to DataFrame
    dataframe = pd.read_excel(data_path + data_file)
    
    return dataframe

def save_model(model, model_path, model_name, container_url, key):
    ''' First, serialize pandas object with model to local disk in 
        pickle file,  then, upload this file to Azure Blob Storage
    '''
    # write model to local disk
    model.to_pickle(model_path + model_name)
    
    # instanciate containder client
    container_client = ContainerClient.from_container_url(
                                        container_url=container_url,
                                        credential=key)
    # create a blob client to the file
    blob_client = container_client.get_blob_client(blob = model_path 
                                                          + model_name)
    
    # first checks if file already exists (same name) before upload it
    if not blob_client.exists():
        with open (model_path + model_name, 'rb') as my_file:
            blob_client.upload_blob(my_file)
            
    # close handles
    blob_client.close()
    container_client.close()
    
    # print a confirmation as we don't return anything
    print('Model upload completed!')

def run_train(data_path, data_file, model_path, model_name, 
                                                    container_url, key):
    '''Run the complete training pipeline
    '''
    # downloads and imports input data
    auto_df = read_input_data(data_path, data_file, container_url, key)
    # create calculated features
    auto_df['cylinder_displacement'] = (auto_df['cylinders'] 
                                        * auto_df['displacement'])
    auto_df['specific_torque'] = (auto_df['horsepower'] 
                                  * auto_df['cylinder_displacement'])
    auto_df['fake_torque'] = auto_df['weight'] / auto_df['acceleration']
    
    #define feature types
    target = 'mpg' # Milhas por galão
    num_vars = ['cylinders', 'displacement', 'horsepower', 'weight', 
                'acceleration', 'year', 'cylinder_displacement', 
                'specific_torque', 'fake_torque']
    cat_vars = ['origin']

    auto_df[cat_vars] = auto_df[cat_vars].astype(str)
    
    # split train and test subsets
    X_train, X_test, y_train, y_test = model_selection.train_test_split( 
                                            auto_df[num_vars+cat_vars],
                                            auto_df[target],
                                            random_state=1992,
                                            test_size=0.25)
    # setup pipeline
    ## Define o transformador do transformação logaritmica
    log = vt.LogTransformer(variables=num_vars) 
    ## Cria Dummys
    onehot = ce.OneHotCategoricalEncoder(variables=cat_vars, 
                                            drop_last=True) 
    model = linear_model.Lasso() # Definição do modelo

    full_pipeline = Pipeline( steps=[
        ("log", log),
        ("onehot", onehot),
        ('model', model) ] )

    param_grid = { 'model__alpha':[0.0167, 0.0001, 0.001, 0.01, 0.1, 
                                    0.2, 0.5, 0.8, 1], # linspace
                   'model__normalize':[True],
                   'model__random_state':[1992]}

    search = model_selection.GridSearchCV(
                                full_pipeline,
                                param_grid,
                                cv=5,
                                n_jobs=-1,
                                scoring='neg_root_mean_squared_error')

    search.fit(X_train, y_train) # Executa o treinamento!!

    best_model = search.best_estimator_
    
     # save results data
    cv_result = pd.DataFrame(search.cv_results_) # Pega resultdos do grid
    cv_result = cv_result.sort_values(by='mean_test_score', 
                                    ascending = False)
    
    # assess model over test data
    y_test_pred = best_model.predict(X_test)
    root_mean_squadred_erro = (
             metrics.mean_squared_error( y_test, y_test_pred) ** (1/2))
    
    # create pandas objet with model
    model_s = pd.Series( {"cat_vars":cat_vars,
                      "num_vars":num_vars,
                      "fit_vars": X_train.columns.tolist(),
                      "model":best_model,
                      "rmse":root_mean_squadred_erro} )
    
    # save model
    save_model(model_s, model_path, model_name, container_url, key)

if __name__ == '__main__':
    run_train(data_path, data_file, model_path, model_name, 
                                                    container_url, key)

    elapsed_time = datetime.datetime.now() - start_time
    print(f'Full process run in: {elapsed_time.total_seconds()} segundos')                                               