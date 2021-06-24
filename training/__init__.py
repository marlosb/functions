import datetime
import logging
import os

import azure.functions as func

from train import run_train

key = os.environ.get('STORAGE_ACCOUNT_KEY')
container_url = 'https://asnrocks.blob.core.windows.net/auto'

data_path = 'data/'
data_file = 'auto.xlsx'

model_path = 'models/'

def main(mytimer: func.TimerRequest) -> None:
    start_time = datetime.datetime.now()
    model_name = (
    f'auto_model_{start_time.strftime("%Y-%m-%d--%H-%M-%S")}.pickle')
    
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', 
                  utc_timestamp)

    run_train(data_path, data_file, model_path, model_name, 
                                                    container_url, key)
    
    elapsed_time = datetime.datetime.now() - start_time
    logging.info(f'Full process run in: {elapsed_time.total_seconds()} \
                                                              segundos')


