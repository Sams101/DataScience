# Transformer Chatbot

### Objective:
- 





### Links

- https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
- set up sql microsoft on docker https://getadigital.com/blog/setting-up-sql-server-on-docker-in-mac-os/
- set up hadoop on docker https://amitasviper.github.io/2018/04/24/running-hadoop-in-docker-on-mac.html
- set up ubuntu on docker https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04
- set up tensflow on docker https://www.tensorflow.org/tfx/serving/serving_advanced
- install fastapi
- install uvicorn


Iris:
https://medium.com/@dvelsner/deploying-a-simple-machine-learning-model-in-a-modern-web-application-flask-angular-docker-a657db075280
https://towardsdatascience.com/deploying-iris-classifications-with-fastapi-and-docker-7c9b83fdec3a


- Good docker tutorial https://docker-curriculum.com/


```python
from sklearn import svm
from sklearn import datasets

# read iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# fit model
clf = svm.SVC(
    C=1.0,
    probability=True, 
    random_state=1)
clf.fit(X, y)

print('Training Accuracy: ', clf.score(X, y))
print('Prediction results: ', clf.predict_proba([[5.2,  3.5,  2.4,  1.2]]))
```

    Training Accuracy:  0.9733333333333334
    Prediction results:  [[0.62204997 0.3382061  0.03974394]]



```python
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from sklearn import datasets

def main():
    clf = LogisticRegression()
    p = Pipeline([('clf', clf)])
    print('Training model...')
    p.fit(X, y)
    print('Model trained!')

    filename_p = 'IrisClassifier.sav'
    print('Saving model in %s' % filename_p)
    joblib.dump(p, filename_p)
    print('Model saved!')

if __name__ == "__main__":
    print('Loading iris data set...')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print('Dataset loaded!')
    main()
    
    
```

    Loading iris data set...
    Dataset loaded!
    Training model...
    Model trained!
    Saving model in IrisClassifier.sav
    Model saved!


    /Users/samtreacy/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)



```python
import docker
client = docker.from_env()
```


```python
client.containers.run("ubuntu", "echo hello world")
```




    b'hello world\n'




```python
client.containers.run("bfirsh/reticulate-splines", detach=True)
```




    <Container: 23cf8b88c3>




```python
client.containers.list()
```




    [<Container: 528bc54b4f>, <Container: cc6b5a0f72>, <Container: 8be8195473>]




```python
client.images.list()
```




    [<Image: 'ubuntu:latest'>,
     <Image: 'mcr.microsoft.com/mssql/server:2019-latest'>,
     <Image: 'python:latest'>,
     <Image: 'postgres:latest'>,
     <Image: 'mysql:latest'>,
     <Image: 'dorowu/ubuntu-desktop-lxde-vnc:latest'>,
     <Image: 'opencpu/ubuntu-20.04:latest'>,
     <Image: 'dorowu/ubuntu-desktop-lxde-vnc:bionic'>,
     <Image: 'bfirsh/reticulate-splines:latest'>,
     <Image: 'sequenceiq/hadoop-docker:2.7.1'>]



# test


```python
client.images.list()
```




    [<Image: 'ubuntu:latest'>,
     <Image: 'mcr.microsoft.com/mssql/server:2019-latest'>,
     <Image: 'python:latest'>,
     <Image: 'postgres:latest'>,
     <Image: 'mysql:latest'>,
     <Image: 'phpmyadmin/phpmyadmin:latest'>,
     <Image: 'dorowu/ubuntu-desktop-lxde-vnc:latest'>,
     <Image: 'opencpu/ubuntu-20.04:latest'>,
     <Image: 'mysql/mysql-server:latest'>,
     <Image: 'mcr.microsoft.com/mssql/server:2017-latest'>,
     <Image: 'dorowu/ubuntu-desktop-lxde-vnc:bionic'>,
     <Image: 'mcr.microsoft.com/mssql/server:2017-CU8-ubuntu'>,
     <Image: 'bfirsh/reticulate-splines:latest'>,
     <Image: 'sequenceiq/hadoop-docker:2.7.1'>]




```python
from flask import Flask, jsonify, request
from fastai.text import *
import json

```


```python
import os

from flask import Flask
from flask_restful import Resource, Api, reqparse
from joblib import load
import numpy as np

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

print("Loading model from: {}".format(MODEL_PATH))
clf = load(MODEL_PATH)

app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def __init__(self):
        self._required_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                                   'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                                   'LSTAT']
        self.reqparse = reqparse.RequestParser()
        for feature in self._required_features:
            self.reqparse.add_argument(
                feature, type = float, required = True, location = 'json',
                help = 'No {} provided'.format(feature))
        super(Prediction, self).__init__()

    def post(self):
        args = self.reqparse.parse_args()
        X = np.array([args[f] for f in self._required_features]).reshape(1, -1)
        y_pred = clf.predict(X)
        return {'prediction': y_pred.tolist()[0]}

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-95-9d44168bc1f7> in <module>
          6 import numpy as np
          7 
    ----> 8 MODEL_DIR = os.environ["MODEL_DIR"]
          9 MODEL_FILE = os.environ["MODEL_FILE"]
         10 METADATA_FILE = os.environ["METADATA_FILE"]


    ~/opt/anaconda3/lib/python3.7/os.py in __getitem__(self, key)
        679         except KeyError:
        680             # raise KeyError with the original key value
    --> 681             raise KeyError(key) from None
        682         return self.decodevalue(value)
        683 


    KeyError: 'MODEL_DIR'



```python

import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from sklearn import datasets

def main():
    clf = LogisticRegression()
    p = Pipeline([('clf', clf)])
    print('Training model...')
    p.fit(X, y)
    print('Model trained!')

    filename_p = 'IrisClassifier.sav'
    print('Saving model in %s' % filename_p)
    joblib.dump(p, filename_p)
    print('Model saved!')

if __name__ == "__main__":
    print('Loading iris data set...')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print('Dataset loaded!')
    main()
```

    Loading iris data set...
    Dataset loaded!
    Training model...
    Model trained!
    Saving model in IrisClassifier.sav
    Model saved!


    /Users/samtreacy/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)



```python
!s2i build . seldonio/seldon-core-s2i-python37:0.13 sklearn-iris:0.1

```

    zsh:1: command not found: s2i



```python
# https://github.com/bentoml/BentoML/blob/master/guides/quick-start/iris_classifier.py
import bentoml
from bentoml.adapters import DataframeInput
from bentoml.artifact import SklearnModelArtifact

@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([SklearnModelArtifact('model')])
class IrisClassifier(bentoml.BentoService):

    @bentoml.api(input=DataframeInput())
    def predict(self, df):
        # Optional pre-processing, post-processing code goes here
        return self.artifacts.model.predict(df)
```


```python
from sklearn import datasets

from iris_classifier import IrisClassifier

if __name__ == "__main__":
    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Model Training
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

    # Create a iris classifier service instance
    iris_classifier_service = IrisClassifier()

    # Pack the newly trained model artifact
    iris_classifier_service.pack('model', clf)

    # Save the prediction service to disk for model serving
    saved_path = iris_classifier_service.save()
```

    [2020-07-01 19:33:12,306] INFO - BentoService bundle 'IrisClassifier:20200701193218_C5D6A2' saved to: /Users/samtreacy/bentoml/repository/IrisClassifier/20200701193218_C5D6A2



```python
# Assuming JQ(https://stedolan.github.io/jq/) was installed, you can also manually
# copy the uri field in `bentoml get` command's JSON output
saved_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")

bentoml serve $saved_path
```


      File "<ipython-input-104-2b2c895fe3ec>", line 3
        saved_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")
                   ^
    SyntaxError: invalid syntax




```python
import requests
response = requests.post("http://127.0.0.1:5000/predict", json=[[5.1, 3.5, 1.4, 0.2]])
print(response.text)
```


    ---------------------------------------------------------------------------

    ConnectionRefusedError                    Traceback (most recent call last)

    ~/opt/anaconda3/lib/python3.7/site-packages/urllib3/connection.py in _new_conn(self)
        159             conn = connection.create_connection(
    --> 160                 (self._dns_host, self.port), self.timeout, **extra_kw
        161             )


    ~/opt/anaconda3/lib/python3.7/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         83     if err is not None:
    ---> 84         raise err
         85 


    ~/opt/anaconda3/lib/python3.7/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         73                 sock.bind(source_address)
    ---> 74             sock.connect(sa)
         75             return sock


    ConnectionRefusedError: [Errno 61] Connection refused

    
    During handling of the above exception, another exception occurred:


    NewConnectionError                        Traceback (most recent call last)

    ~/opt/anaconda3/lib/python3.7/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        676                 headers=headers,
    --> 677                 chunked=chunked,
        678             )


    ~/opt/anaconda3/lib/python3.7/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        391         else:
    --> 392             conn.request(method, url, **httplib_request_kw)
        393 


    ~/opt/anaconda3/lib/python3.7/http/client.py in request(self, method, url, body, headers, encode_chunked)
       1251         """Send a complete request to the server."""
    -> 1252         self._send_request(method, url, body, headers, encode_chunked)
       1253 


    ~/opt/anaconda3/lib/python3.7/http/client.py in _send_request(self, method, url, body, headers, encode_chunked)
       1297             body = _encode(body, 'body')
    -> 1298         self.endheaders(body, encode_chunked=encode_chunked)
       1299 


    ~/opt/anaconda3/lib/python3.7/http/client.py in endheaders(self, message_body, encode_chunked)
       1246             raise CannotSendHeader()
    -> 1247         self._send_output(message_body, encode_chunked=encode_chunked)
       1248 


    ~/opt/anaconda3/lib/python3.7/http/client.py in _send_output(self, message_body, encode_chunked)
       1025         del self._buffer[:]
    -> 1026         self.send(msg)
       1027 


    ~/opt/anaconda3/lib/python3.7/http/client.py in send(self, data)
        965             if self.auto_open:
    --> 966                 self.connect()
        967             else:


    ~/opt/anaconda3/lib/python3.7/site-packages/urllib3/connection.py in connect(self)
        186     def connect(self):
    --> 187         conn = self._new_conn()
        188         self._prepare_conn(conn)


    ~/opt/anaconda3/lib/python3.7/site-packages/urllib3/connection.py in _new_conn(self)
        171             raise NewConnectionError(
    --> 172                 self, "Failed to establish a new connection: %s" % e
        173             )


    NewConnectionError: <urllib3.connection.HTTPConnection object at 0x7fe3ac9d17d0>: Failed to establish a new connection: [Errno 61] Connection refused

    
    During handling of the above exception, another exception occurred:


    MaxRetryError                             Traceback (most recent call last)

    ~/opt/anaconda3/lib/python3.7/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        448                     retries=self.max_retries,
    --> 449                     timeout=timeout
        450                 )


    ~/opt/anaconda3/lib/python3.7/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        724             retries = retries.increment(
    --> 725                 method, url, error=e, _pool=self, _stacktrace=sys.exc_info()[2]
        726             )


    ~/opt/anaconda3/lib/python3.7/site-packages/urllib3/util/retry.py in increment(self, method, url, response, error, _pool, _stacktrace)
        438         if new_retry.is_exhausted():
    --> 439             raise MaxRetryError(_pool, url, error or ResponseError(cause))
        440 


    MaxRetryError: HTTPConnectionPool(host='127.0.0.1', port=5000): Max retries exceeded with url: /predict (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fe3ac9d17d0>: Failed to establish a new connection: [Errno 61] Connection refused'))

    
    During handling of the above exception, another exception occurred:


    ConnectionError                           Traceback (most recent call last)

    <ipython-input-105-a4e1846a0077> in <module>
          1 import requests
    ----> 2 response = requests.post("http://127.0.0.1:5000/predict", json=[[5.1, 3.5, 1.4, 0.2]])
          3 print(response.text)


    ~/opt/anaconda3/lib/python3.7/site-packages/requests/api.py in post(url, data, json, **kwargs)
        117     """
        118 
    --> 119     return request('post', url, data=data, json=json, **kwargs)
        120 
        121 


    ~/opt/anaconda3/lib/python3.7/site-packages/requests/api.py in request(method, url, **kwargs)
         59     # cases, and look like a memory leak in others.
         60     with sessions.Session() as session:
    ---> 61         return session.request(method=method, url=url, **kwargs)
         62 
         63 


    ~/opt/anaconda3/lib/python3.7/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        528         }
        529         send_kwargs.update(settings)
    --> 530         resp = self.send(prep, **send_kwargs)
        531 
        532         return resp


    ~/opt/anaconda3/lib/python3.7/site-packages/requests/sessions.py in send(self, request, **kwargs)
        641 
        642         # Send the request
    --> 643         r = adapter.send(request, **kwargs)
        644 
        645         # Total elapsed time of the request (approximately)


    ~/opt/anaconda3/lib/python3.7/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        514                 raise SSLError(e, request=request)
        515 
    --> 516             raise ConnectionError(e, request=request)
        517 
        518         except ClosedPoolError as e:


    ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=5000): Max retries exceeded with url: /predict (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fe3ac9d17d0>: Failed to establish a new connection: [Errno 61] Connection refused'))



```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

data = pd.read_csv("kc_house_data.csv")

labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)

col_imp = ["grade", "lat", "long", "sqft_living", "waterfront", "yr_built"]

clf = GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2)
clf.fit(train1[col_imp], labels)

def predict(dict_values, col_imp=col_imp, clf=clf):
    x = np.array([float(dict_values[col]) for col in col_imp])
    x = x.reshape(1,-1)
    y_pred = clf.predict(x)[0]
    return y_pred
```


```python
import json
import pickle
import numpy as np
from flask import Flask, request
# 

flask_app = Flask(__name__)

#ML model path
model_path = "ML_Model/model.pkl"


@flask_app.route('/', methods=['GET'])
def index_page():
    return_data = {
        "error" : "0",
        "message" : "Successful"
    }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')

@flask_app.route('/predict',methods=['GET'])
def model_deploy():
    try:
        age = request.form.get('age')
        bs_fast = request.form.get('BS_Fast')
        bs_pp = request.form.get('BS_pp')
        plasma_r = request.form.get('Plasma_R')
        plasma_f = request.form.get('Plasma_F')
        HbA1c = request.form.get('HbA1c')
        fields = [age,bs_fast,bs_pp,plasma_r,plasma_f,HbA1c]
        if not None in fields:
            #Datapreprocessing Convert the values to float
            age = float(age)
            bs_fast = float(bs_fast)
            bs_pp = float(bs_pp)
            plasma_r = float(plasma_r)
            plasma_f = float(plasma_f)
            hbA1c = float(HbA1c)
            result = [age,bs_fast,bs_pp,plasma_r,plasma_f,HbA1c]
            #Passing data to model & loading the model from disk
            classifier = pickle.load(open(model_path, 'rb'))
            prediction = classifier.predict([result])[0]
            conf_score =  np.max(classifier.predict_proba([result]))*100
            return_data = {
                "error" : '0',
                "message" : 'Successfull',
                "prediction": prediction,
                "confidence_score" : conf_score.round(2)
            }
        else:
            return_data = {
                "error" : '1',
                "message": "Invalid Parameters"             
            }
    except Exception as e:
        return_data = {
            'error' : '2',
            "message": str(e)
            }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')


if __name__ == "__main__":
    flask_app.run(host ='0.0.0.0',port=8080, debug=False)
```

     * Serving Flask app "__main__" (lazy loading)
     * Environment: production
    [31m   WARNING: This is a development server. Do not use it in a production deployment.[0m
    [2m   Use a production WSGI server instead.[0m
     * Debug mode: off



    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-110-e2ef642f5f74> in <module>
         62 
         63 if __name__ == "__main__":
    ---> 64     flask_app.run(host ='0.0.0.0',port=8080, debug=False)
    

    ~/opt/anaconda3/lib/python3.7/site-packages/flask/app.py in run(self, host, port, debug, load_dotenv, **options)
        988 
        989         try:
    --> 990             run_simple(host, port, self, **options)
        991         finally:
        992             # reset the first request information if the development server


    ~/opt/anaconda3/lib/python3.7/site-packages/werkzeug/serving.py in run_simple(hostname, port, application, use_reloader, use_debugger, use_evalex, extra_files, reloader_interval, reloader_type, threaded, processes, request_handler, static_files, passthrough_errors, ssl_context)
       1050         run_with_reloader(inner, extra_files, reloader_interval, reloader_type)
       1051     else:
    -> 1052         inner()
       1053 
       1054 


    ~/opt/anaconda3/lib/python3.7/site-packages/werkzeug/serving.py in inner()
       1003             passthrough_errors,
       1004             ssl_context,
    -> 1005             fd=fd,
       1006         )
       1007         if fd is None:


    ~/opt/anaconda3/lib/python3.7/site-packages/werkzeug/serving.py in make_server(host, port, app, threaded, processes, request_handler, passthrough_errors, ssl_context, fd)
        846     elif threaded:
        847         return ThreadedWSGIServer(
    --> 848             host, port, app, request_handler, passthrough_errors, ssl_context, fd=fd
        849         )
        850     elif processes > 1:


    ~/opt/anaconda3/lib/python3.7/site-packages/werkzeug/serving.py in __init__(self, host, port, app, handler, passthrough_errors, ssl_context, fd)
        750             self.socket.close()
        751             self.socket = real_sock
    --> 752             self.server_address = self.socket.getsockname()
        753 
        754         if ssl_context is not None:


    OSError: [Errno 102] Operation not supported on socket



```python

```
