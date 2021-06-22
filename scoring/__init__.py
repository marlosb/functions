import datetime
import json
import logging
import os

import azure.functions as func

from score import get_model, get_score

key = os.environ.get('STORAGE_ACCOUNT_KEY')
segredo = os.environ.get('segredo')
container_url = 'https://asnrocks.blob.core.windows.net/auto'

model_path = 'models/'

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    start_time = datetime.datetime.now()

    try:
        input_dict = req.get_json()
    except ValueError:
        return func.HttpResponse("No input data!", status_code=200)

    model_package, model_version = get_model(model_path, container_url, 
                                            key)
    try:                                        
        score = get_score(input_dict, model_package)
    except KeyError:
        return func.HttpResponse("Wrong input parameters", 
                                 status_code=200)

    headers_list = [i for i in req.headers.keys()]

    if 'asnrockskey' not in headers_list:
        return func.HttpResponse("Access Denied!", status_code=401)

    if req.headers.get('asnrockskey') != segredo:
        return func.HttpResponse("Access Denied!", status_code=403)

    elapsed_time = datetime.datetime.now() - start_time
    results_dict = {
                'Start time': start_time.strftime('%Y-%m-%d--%H-%M-%S'),
                'model version': model_version,
                'input data': input_dict,
                'predicted score': score[0],
                'scoring time': elapsed_time.total_seconds()}

    return func.HttpResponse(
            json.dumps(results_dict),
            status_code=200,
            mimetype='application/json'
        )