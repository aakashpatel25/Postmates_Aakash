# Hand Written Digit Recognition System
##### By Aakash Moghariya
##### Postmates

This repository contains program to train a neural network model on MNIST hand written dataset using Keras. The trained network is then freezed and served as a tensorflow model using flask-restplus framework. 


##### Technology Used
 - Python 2.7
 - Keras (With tensorflow as backend)
 - Tensorflow 
 - Flask-restplus
 - HTML
 - JQuery
 

##### Deployment instruction
1. For training and saving a model MNIST_Network.py is used, example for running the code are illustrated below. Default value number of epoch is 10, default batch_size value is 200 and efault value for the model name is handwritten_digit_recog_model.h5 
    ```Bash
    python MNIST_Network.py
    python MNIST_Network.py --epoch 100 --batch_size 200 --model_name handwritten_digit_recog_model.h5
    ```
2. To freeze the trained model use freeze.py. Default for the saved keras model is specified above, frozen model default would be 'digit_model.pb' and default path for the saving frozen model is the current working directory. The program could be executed as follows
    ```Bash
    python freeze.py
    python freeze.py --model_name digit_model1.h5 --frozen_model_name digit.pb --frozen_model_path /home/ubuntu
    ```
3. Copy the forzen model to ../restplus/api/model/model_file/
4. To run rest api to serve tesnorflow model go to ../restplus/settings.py modify the host name and other settings as needed. Then modify the model arguments at ../restplus/api/model/modelconfig.py
    ```bash
    python restplus/app.py
    ```
5. Swagger UI is delpoyed at http://hostname:port. It can be used by developers to develop and test APIs by means of sending sample request to the API and receiving the response. 
6. Ensure corss-origin-request is turned on in your browser.
7. Modify the request url in the ajax request of index.html. Run index.html to test the app.

**Note**: To view the model architecture of the trained model open model_plot.png

##### Pretrained Model Download
To download a pretrained keras as well as frozen mode use the google drive link specified below. 
   
    [Pre Trained Models](https://drive.google.com/drive/folders/1wGDci2gTGqHuya4n3eLRphNrIC-Yv-CF?usp=sharing)
