# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##########

import time

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import summary
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import summary_io
from tensorflow.python.training import supervisor
from tensorflow.python.training import training_util

###########

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

import numpy as np



slim = tf.contrib.slim


big_labels = []
big_predictions =  []
big_logits = []

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


#####################


def evaluation(sess,
               num_evals=1,
               initial_op=None,
               initial_op_feed_dict=None,
               eval_op=None,
               eval_op_feed_dict=None,
               final_op=None,
               final_op_feed_dict=None,
               summary_op=None,
               summary_op_feed_dict=None,
               summary_writer=None,
               global_step=None,
               cm = None):
  """Performs a single evaluation run.
  A single evaluation consists of several steps run in the following order:
  (1) an initialization op, (2) an evaluation op which is executed `num_evals`
  times (3) a finalization op and (4) the execution of a summary op which is
  written out using a summary writer.
  Args:
    sess: The current TensorFlow `Session`.
    num_evals: The number of times to execute `eval_op`.
    initial_op: An operation run at the beginning of evaluation.
    initial_op_feed_dict: A feed dictionary to use when executing `initial_op`.
    eval_op: A operation run `num_evals` times.
    eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
    final_op: An operation to execute after all of the `eval_op` executions. The
      value of `final_op` is returned.
    final_op_feed_dict: A feed dictionary to use when executing `final_op`.
    summary_op: A summary op executed after `eval_op` and `finalize_op`.
    summary_op_feed_dict: An optional feed dictionary to use when executing the
      `summary_op`.
    summary_writer: The summery writer used if `summary_op` is provided.
    global_step: the global step variable. If left as `None`, then
      slim.variables.global_step() is used.
  Returns:
    The value of `final_op` or `None` if `final_op` is `None`.
  Raises:
    ValueError: if `summary_op` is provided but `global_step` is `None`.
  """

  #big_confusion_matrix = np.zeros([10,10])
  accurancy = 0

  if initial_op is not None:
    logging.info('Executing initial eval op')
    sess.run(initial_op, initial_op_feed_dict)

  if eval_op is not None:
    logging.info('Executing eval ops')
    for i in range(int(num_evals)):
      logging.info('Executing eval_op %d/%d', i + 1, num_evals)
      accurancy += float(sess.run(eval_op, eval_op_feed_dict)[1])
      #confusion_matrix = cm[0]
      #big_confusion_matrix = big_confusion_matrix + np.array(sess.run(confusion_matrix))

#      big_labels.append(np.array(sess.run(cm[1])))
#      big_predictions.append(np.array(sess.run(cm[2])))
#      big_logits.append(np.array(sess.run(cm[3])))
      #logging.info(sess.run(cm))


 # with open("confusion_matrix.txt", "a") as myfile:
  #  myfile.write(str(big_confusion_matrix.flatten()))
 #   myfile.write('\n')
#  myfile.close()
  
#  with open("labels.txt", "a") as myfile:
#    myfile.write(str(big_labels))
#    myfile.write('\n')
#  myfile.close()
  
#  with open("predictions.txt", "a") as myfile:
#    myfile.write(str(big_predictions))
#    myfile.write('\n')
#  myfile.close()
  
#  with open("logits.txt", "a") as myfile:
#    myfile.write(str(big_logits))
#    myfile.write('\n')
#  myfile.close()


  if final_op is not None:
    logging.info('Executing final op')
    final_op_value = sess.run(final_op, final_op_feed_dict)
  else:
    final_op_value = None

  if summary_op is not None:
    logging.info('Executing summary op')
    if global_step is None:
      global_step = variables.get_or_create_global_step()

    global_step = training_util.global_step(sess, global_step)
    summary_str = sess.run(summary_op, summary_op_feed_dict)
    accurancy = accurancy/float(int(num_evals))
    accfile = open('accurancies.txt','a')
    accfile.write(str(accurancy))
    accfile.write(',')
    accfile.close()
    #np.save('confusion_matrix',big_confusion_matrix)
    summary_writer.add_summary(summary_str, global_step)
    summary_writer.flush()

  return final_op_value


_USE_DEFAULT = 0


def evaluate_once(master,
                  checkpoint_path,
                  logdir,
                  num_evals=1,
                  initial_op=None,
                  initial_op_feed_dict=None,
                  eval_op=None,
                  eval_op_feed_dict=None,
                  final_op=None,
                  final_op_feed_dict=None,
                  summary_op=_USE_DEFAULT,
                  summary_op_feed_dict=None,
                  variables_to_restore=None,
                  session_config=None,
                  cm = None):
  """Evaluates the model at the given checkpoint path.
  Args:
    master: The BNS address of the TensorFlow master.
    checkpoint_path: The path to a checkpoint to use for evaluation.
    logdir: The directory where the TensorFlow summaries are written to.
    num_evals: The number of times to run `eval_op`.
    initial_op: An operation run at the beginning of evaluation.
    initial_op_feed_dict: A feed dictionary to use when executing `initial_op`.
    eval_op: A operation run `num_evals` times.
    eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
    final_op: An operation to execute after all of the `eval_op` executions. The
      value of `final_op` is returned.
    final_op_feed_dict: A feed dictionary to use when executing `final_op`.
    summary_op: The summary_op to evaluate after running TF-Slims metric ops. By
      default the summary_op is set to tf.summary.merge_all().
    summary_op_feed_dict: An optional feed dictionary to use when running the
      `summary_op`.
    variables_to_restore: A list of TensorFlow variables to restore during
      evaluation. If the argument is left as `None` then
      slim.variables.GetVariablesToRestore() is used.
    session_config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
  Returns:
    The value of `final_op` or `None` if `final_op` is `None`.
  """


  if summary_op == _USE_DEFAULT:
    summary_op = tf.merge_all_summaries()

  global_step = variables.get_or_create_global_step()

  saver = tf_saver.Saver( variables_to_restore or variables.get_variables_to_restore())

  summary_writer = summary_io.SummaryWriter(logdir)

  sv = supervisor.Supervisor(graph=ops.get_default_graph(),
                             logdir=logdir,
                             summary_op=None,
                             summary_writer=None,
                             global_step=None,
                             saver=None)

  logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                         time.gmtime()))
  with sv.managed_session(
      master, start_standard_services=False, config=session_config) as sess:
    saver.restore(sess, checkpoint_path)
    sv.start_queue_runners(sess)
    final_op_value = evaluation(sess,
                                num_evals=num_evals,
                                initial_op=initial_op,
                                initial_op_feed_dict=initial_op_feed_dict,
                                eval_op=eval_op,
                                eval_op_feed_dict=eval_op_feed_dict,
                                final_op=final_op,
                                final_op_feed_dict=final_op_feed_dict,
                                summary_op=summary_op,
                                summary_op_feed_dict=summary_op_feed_dict,
                                summary_writer=summary_writer,
                                global_step=global_step,
                                cm = cm)

  logging.info('Finished evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                         time.gmtime()))

  return final_op_value


####################


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    num_classes=(dataset.num_classes - FLAGS.labels_offset)
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)
    print('numclasses')
    print(num_classes)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)



    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall@5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.iteritems():
      summary_name = 'eval/%s' % name
      op = tf.scalar_summary(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)


    confusion_matrix = tf.contrib.metrics.confusion_matrix(predictions,labels)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    cms = [confusion_matrix,labels,predictions,logits]

    evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        variables_to_restore=variables_to_restore)

    tf.logging.info(predictions)
    tf.logging.info(labels)


if __name__ == '__main__':
  tf.app.run()
