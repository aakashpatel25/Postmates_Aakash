from keras import backend as K
from keras.models import load_model
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        tf.ConfigProto(allow_soft_placement=True)
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def main(trained_model, frozed_model, forzen_model_path):
    model = load_model(trained_model)
    frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, forzen_model_path, frozed_model, as_text=False)


if __name__=="__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description = 'MNIST Neural Network Training Argument Parser')
    parser.add_argument('--model_name', type=str, default='handwritten_digit_recog_model.h5' ,help='Enter the saved model file name.')
    parser.add_argument('--frozen_model_name', type=str, default='digit_model.pb' ,help='Enter the final frozen file name.')
    parser.add_argument('--frozen_model_path', type=str, default=os.getcwd() ,help='Enter the path where frozen model is to be saved.')
    args = parser.parse_args()
    main(args.model_name, args.frozen_model_name, args.frozen_model_path)