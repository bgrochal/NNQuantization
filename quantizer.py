import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import bin_quantization
import fixed_point_quantization
from models.research.slim.nets import inception


def perform_quantization(checkpoint_file_path, classes, quantization_algorithm, quantization_bits):
    """
    Loads the TensorFlow Slim model and variables contained in the checkpoint file, and then performs a quantization and
    dequantization of trainable variables (i.e. weights and batch norms). Finally, new values of the trainable variables
    are saved in a checkpoint file, which may be used to perform inference with it.
    """

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()  # needed by the Saver mechanism

        # MODEL CREATION
        image_size = inception.inception_v1.default_image_size

        with slim.arg_scope(inception.inception_v1_arg_scope()):
            inception.inception_v1(np.zeros((64, image_size, image_size, 3), np.float32),
                                   num_classes=classes, is_training=False)
        init_function = slim.assign_from_checkpoint_fn(checkpoint_file_path, slim.get_model_variables())

        with tf.Session() as session:
            # INITIALIZATION
            init_function(session)
            session.run(global_step.initializer)

            # GETTING ALL TRAINABLE VARIABLES - WEIGHTS AND BATCH NORMS
            variables = tf.trainable_variables()

            # RECORDING QUANTIZATION ERRORS
            quantization_errors = []

            # SAVER OPTIONS
            checkpoint_saver = tf.train.Saver()
            filename_position = checkpoint_file_path.find(checkpoint_file_path.rsplit('/')[-1])

            # BIN QUANTIZATION ALGORITHM
            if quantization_algorithm == 1:
                for variable_number, variable in enumerate(variables):
                    values = session.run(variable)

                    # BIN QUANTIZATION
                    quantized_data, min_value, max_value = bin_quantization \
                        .quantize(values.reshape(-1), quantization_bits)

                    # BIN DEQUANTIZATION
                    dequantized_data = bin_quantization \
                        .dequantize(quantized_data, quantization_bits, min_value, max_value) \
                        .reshape(values.shape)

                    # VARIABLE UPDATE and ERRORS SAVING
                    session.run(variable.assign(dequantized_data))
                    quantization_errors.append(calculate_relative_error(dequantized_data, values))

                    # LOGGING
                    if (variable_number + 1) % 20 == 0 or (variable_number + 1) == len(variables):
                        print('### Processed variables: {}/{}'.format(variable_number + 1, len(variables)))

                # SAVING UPDATED MODEL
                output_file_path = checkpoint_file_path[:filename_position] + 'bin_{}bits/'.format(quantization_bits) + \
                                   checkpoint_file_path[filename_position:]

                saving_path = checkpoint_saver.save(session, output_file_path, global_step=global_step)
                print('Quantized variables saved under: {}\n'.format(saving_path))

            # FIXED-POINT QUANTIZATION ALGORITHM
            elif quantization_algorithm == 2:
                for variable_number, variable in enumerate(variables):
                    values = session.run(variable)

                    # FIXED-POINT QUANTIZATION
                    quantized_data, shift_positions, fractional_part_width, has_sign = fixed_point_quantization \
                        .quantize(values.reshape(-1), quantization_bits)

                    # FIXED-POINT DEQUANTIZATION
                    dequantized_data = fixed_point_quantization \
                        .dequantize(quantized_data, shift_positions, fractional_part_width, has_sign) \
                        .reshape(values.shape)

                    # VARIABLE UPDATE and ERRORS SAVING
                    session.run(variable.assign(dequantized_data))
                    quantization_errors.append(calculate_relative_error(dequantized_data, values))

                    # LOGGING
                    if (variable_number + 1) % 20 == 0 or (variable_number + 1) == len(variables):
                        print('### Processed variables: {}/{}'.format(variable_number + 1, len(variables)))

                # SAVING UPDATED MODEL
                output_file_path = checkpoint_file_path[:filename_position] + \
                                   'fixed-point_{}bits/'.format(quantization_bits) + \
                                   checkpoint_file_path[filename_position:]

                saving_path = checkpoint_saver.save(session, output_file_path, global_step=global_step)
                print('Quantized variables saved under: {}\n'.format(saving_path))

            # NOT RECOGNIZED ALGORITHM
            else:
                raise Exception('Quantization algorithm type not recognized.')

            # ERRORS SUMMARY
            quantization_errors = np.array(quantization_errors)
            print('Min error: {:.2f}\nMax error: {:.2f}\nAvg error: {:.2f}'.format(quantization_errors.min(),
                                                                                   quantization_errors.max(),
                                                                                   quantization_errors.mean()))


def calculate_relative_error(calculated, reference):
    return np.linalg.norm(calculated - reference) / np.linalg.norm(reference) * 100


def main():
    perform_quantization(checkpoint_file_path=sys.argv[1], classes=int(sys.argv[2]),
                         quantization_algorithm=int(sys.argv[3]), quantization_bits=int(sys.argv[4]))


if __name__ == '__main__':
    main()
