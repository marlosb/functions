import datetime
import logging
import os

import azure.functions as func

from train import run_train

key = os.environ.get('STORAGE_ACCOUNT_KEY')
container_url = 'https://asnrocks.blob.core.windows.net/auto'


model_path = 'models/'

def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")

    start_time = datetime.datetime.now()
    model_name = (
    f'storage_model_{start_time.strftime("%Y-%m-%d--%H-%M-%S")}.pickle')

    data_file = myblob.name.split('/')[-1]
    data_path = myblob.name.split('/')[-2] + '/'

    run_train(data_path, data_file, model_path, model_name, 
                                                    container_url, key)
    
    elapsed_time = datetime.datetime.now() - start_time
    logging.info(f'Full process run in: {elapsed_time.total_seconds()} \
                                                              segundos')