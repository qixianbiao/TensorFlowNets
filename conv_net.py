import tensorflow.python.platform
import tensorflow as tf
import re

from PIL import Image

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 30,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/conv_net_data',
                           """Path to the CIFAR-10 data directory.""")

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


class TensorConv(object):

    def __init__(self):
        pass

    def _activation_summary(self, x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.
        Args:
        x: Tensor
        Returns:
        nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
        Returns:
        Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
        Returns:
        Variable Tensor
        """
        var = self._variable_on_cpu(name, shape,
                     tf.truncated_normal_initializer(stddev=stddev))
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def inference(self, images):
        """Build the CIFAR-10 model.
        Args:
        images: Images returned from distorted_inputs() or inputs().
        Returns:
        Logits.
        """
        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().
        #
        # conv1
        with tf.variable_scope('conv1') as scope:
            kern_size = (5, 5)
            n_kerns   = 3 #first is depth of image
            n_kerns_out = 64

            kernel = self._variable_with_weight_decay('weights', shape=[kern_size[0], kern_size[1], n_kerns, n_kerns_out],
                                                 stddev=1e-4, wd=0.0)

            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            conv1 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv1)
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                             name='norm1')
      # conv2
        with tf.variable_scope('conv2') as scope:
            kern_size = (5, 5)
            n_kerns = 64
            n_kerns_out = 64
            kernel = self._variable_with_weight_decay('weights', shape=[kern_size[0], kern_size[1], n_kerns, n_kerns_out],
                                                 stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            conv2 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv2)

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                             name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        
        # local3
        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            dim = 1
            for d in pool2.get_shape()[1:].as_list():
              dim *= d
            reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])
            weights = self._variable_with_weight_decay('weights', shape=[dim, 384],
                                                  stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
            self._activation_summary(local3)
        
        # local4
        with tf.variable_scope('local4') as scope:
            weights = self._variable_with_weight_decay('weights', shape=[384, 192],
                                                  stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            local4 = tf.nn.relu_layer(local3, weights, biases, name=scope.name)
            self._activation_summary(local4)
        
        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear') as scope:
            weights = self._variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                                  stddev=1/192.0, wd=0.0)
            biases = self._variable_on_cpu('biases', [NUM_CLASSES],
                                      tf.constant_initializer(0.0))
            softmax_linear = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)
            self._activation_summary(softmax_linear)
        
        return softmax_linear            

    def _generate_image_and_label_batch(self, image, label, min_queue_examples):
        """Construct a queued batch of images and labels.
        Args:
        image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
        """
        # Create a queue that shuffles the examples, and then
        # read 'FLAGS.batch_size' images + labels from the example queue.
        num_preprocess_threads = 16
        images, label_batch = tf.train.shuffle_batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * FLAGS.batch_size,
          min_after_dequeue=min_queue_examples)

        # Display the training images in the visualizer.
        tf.image_summary('images', images)
        return images, tf.reshape(label_batch, [FLAGS.batch_size])

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
        Add summary for for "Loss" and "Loss/avg".
        Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
        Returns:
        Loss tensor of type float.
        """
        # Reshape the labels into a dense Tensor of
        # shape [batch_size, NUM_CLASSES].
        sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
        indices = tf.reshape(tf.range(0, FLAGS.batch_size, 1), [FLAGS.batch_size, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        dense_labels = tf.sparse_to_dense(concated,
                                        [FLAGS.batch_size, NUM_CLASSES],
                                        1.0, 0.0)
        # Calculate the average cross entropy loss across the batch.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
          logits, dense_labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')
        
    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CIFAR-10 model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
        total_loss: Total loss from loss().
        Returns:
        loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        # Attach a scalar summmary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(l.op.name +' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))
        return loss_averages_op

    def train(self, total_loss, global_step):
        """Train CIFAR-10 model.
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
        Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
          processed.
        Returns:
        train_op: op for training.
        """
        # Variables that affect learning rate.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                      global_step,
                                      decay_steps,
                                      LEARNING_RATE_DECAY_FACTOR,
                                      staircase=True)
        tf.scalar_summary('learning_rate', lr)
        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(total_loss)
        
        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(total_loss)
        
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        
        # Add histograms for gradients.
        for grad, var in grads:
            if grad:
                tf.histogram_summary(var.op.name + '/gradients', grad)
        
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
          MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        
        return train_op

    def read_from_file(self, image_paths_file):
        class AuxRecord(object):
            pass
        result = AuxRecord()

        # with open(image_paths_file) as fid:            
        #     filenames = []
        #     for line in fid:
        #         line = line.split('\n')
        #         filenames.append(line[0])

        # print filenames
        filename_queue = tf.train.string_input_producer(['train.tfrecords'])
        # print dir(filename_queue)

        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        label_bytes = 1  # 2 for CIFAR-100
        result.height = 32
        result.width = 32
        result.depth = 3
        image_bytes = result.height * result.width * result.depth
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        record_bytes = image_bytes

        # Read a record, getting filenames from the filename_queue.  No
        # header or footer in the CIFAR-10 format, so we leave header_bytes
        # and footer_bytes at their default of 0.
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
          serialized_example,
          dense_keys=['image_raw', 'label'],
          # Defaults are not specified since both keys are required.
          dense_types=[tf.string, tf.int64])

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([32*32*3])

        # OPTIONAL: Could reshape into a NxN image and apply distortions

        image = tf.cast(image, tf.float32)
        label = tf.cast(features['label'], tf.int32)

        image = tf.reshape(image, [32, 32, 3])

        min_queue_examples = 2
        print ('Filling queue with %d images' % min_queue_examples)

        return self._generate_image_and_label_batch(image, label, min_queue_examples)


