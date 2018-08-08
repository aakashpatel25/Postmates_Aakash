from restplus.api.model import modelconfig
from keras.utils import np_utils
import tensorflow as tf, json, logging

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
	return graph

def predict(image_data):
	"""
		Given image data runs tensorflow graph of frozen digit recognizer model
		and returns probabilty.

		@parameter image_data
		@returns json_response
	"""
	y_out = persistent_sess.run(y, feed_dict={
        x: image_data
	})
	json_data = json.dumps({'status': 200,
							'probability': y_out.tolist()})
	return json_data

logger = logging.getLogger(__name__)
logger.info('Loading frozen model graph from {}'.format(modelconfig.FROZEN_MODEL_FILE_NAME))
graph = load_graph(modelconfig.FROZEN_MODEL_FILE_NAME)
logger.info('Finished loading frozen graph and starting tensorflow session')
x = graph.get_tensor_by_name('prefix/input_1:0')
y = graph.get_tensor_by_name('prefix/dense_3/Softmax:0')

persistent_sess = tf.Session(graph=graph)
