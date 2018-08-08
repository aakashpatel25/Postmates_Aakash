from restplus.api.model.digitrecog import predict
from restplus.api.model.parsers import file_upload
from restplus.api.apiInit import api
from flask_restplus import Resource
import logging, cv2, numpy as np, logging
from flask import request
from PIL import Image

logger = logging.getLogger(__name__)

ns = api.namespace('recognize', description='Hello')

@ns.route('/')
class HandWrittenDigitRecognizer(Resource):


    @api.expect(file_upload)
    def post(self):
        """
        	Given an image of handwritten digit returns the probabilty map of 
        	image being a particular digit between 0 to 9.

        	Given a file storage data in request, function preprocesses the data
        	for prediction and then predicts the lable using the forzen digit
        	recognition model.

        	@expects: image
        	@returns: json_probability
        """
	logger.info('Received reqeust to predict a digit')
        img = Image.open(request.files['image']).convert('L')
    	img = np.array(img)
    	img = cv2.resize(img,(28,28))
    	# img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    	img_data = img.reshape(1, 1, 28, 28).astype('float32')
    	logger.info('Served the reqeust by predicting possible probabilities for each digit')
	return predict(img_data)
