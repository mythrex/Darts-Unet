import os
import sys
import time
import glob
import numpy as np
import logging
import argparse
import tensorflow as tf
import numpy as np
import utils
from tqdm import tqdm
import shutil
import time
from scipy.special import softmax

from model_search import Network
from architect_graph import Architect
import utils
from genotypes import Genotype

# config for training
tf.app.flags.DEFINE_float('seed',
                          6969,
                          'Random seed')

# * model related flags
tf.app.flags.DEFINE_float('momentum',
                          0.9,
                          'Momentum')

tf.app.flags.DEFINE_float('weight_decay',
                          3e-4,
                          'Weight Decay')

tf.app.flags.DEFINE_float('arch_learning_rate',
                          3e-1,
                          'Arch LR')

tf.app.flags.DEFINE_float('arch_weight_decay',
                          1e-3,
                          'Arch for weight decay')  # not used till now

tf.app.flags.DEFINE_float('grad_clip',
                          5,
                          'Gradient clip value')

tf.app.flags.DEFINE_float('learning_rate',
                          0.025,
                          'Initial LR for network')

tf.app.flags.DEFINE_float('learning_rate_decay',
                          0.97,
                          'Decay Rate for LR')

tf.app.flags.DEFINE_float('learning_rate_min',
                          0.0001,
                          'Minimum LR')

tf.app.flags.DEFINE_bool('unrolled',
                         True,
                         'if arch search enabled')

tf.app.flags.DEFINE_integer('init_channels',
                            3,
                            'Initial No of channels in Image')

tf.app.flags.DEFINE_integer('num_layers',
                            3,
                            'Decides depth_level of arch as (num_layers // 2)')

tf.app.flags.DEFINE_integer('num_classes',
                            6,
                            'No of classes')

tf.app.flags.DEFINE_list('crop_size',
                         [64, 64],
                         'image size fed to network')


# * app related flags

tf.app.flags.DEFINE_integer('steps',
                            None,
                            'Steps for evaluation.')
tf.app.flags.DEFINE_integer('max_steps',
                            10000,
                            'Max Steps')

tf.app.flags.DEFINE_integer('train_batch_size',
                            8,
                            'Train Batch Size')

tf.app.flags.DEFINE_integer('eval_batch_size',
                            8,
                            'Eval Batch Size')

tf.app.flags.DEFINE_integer('save_checkpoints_steps',
                            100,
                            'Save Checkpoints and perform eval after these many steps')

tf.app.flags.DEFINE_string('model_dir',
                           None,
                           'Model Events and checkpoint folder. Make sure it is a bucket.')

tf.app.flags.DEFINE_string('data',
                           None,
                           'Dataset folder for tf records, better it be bucket')

tf.app.flags.DEFINE_enum('mode',
                         'train',
                         ['train_eval', 'eval', 'train'],
			 'mode=train/eval/train_eval')

tf.app.flags.DEFINE_integer('num_train_examples',
                            2000,
                            'Num Train Examples')

# TPU related flags
tf.app.flags.DEFINE_bool('use_tpu',
                         True,
                         'If to use tpu')

tf.app.flags.DEFINE_bool('use_host_call',
                         True,
                         'If to use host_call_fn')

tf.app.flags.DEFINE_string('tpu',
                           None,
                           'Name of Tpu')
tf.app.flags.DEFINE_string('zone',
                           None,
                           'Name of zone fo your tpu')
tf.app.flags.DEFINE_string('project',
                           None,
                           'Name of Project')

# TPU Config
tf.app.flags.DEFINE_integer('num_shards',
                            8,
                            'Num Shards')
tf.app.flags.DEFINE_integer(
    'iterations_per_loop', default=500,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

tf.app.flags.DEFINE_bool('use_bfloat',
                         False,
                         'if to use bfloat for model')


# train_and_evaluate_config
tf.app.flags.DEFINE_integer('steps_per_eval',
                            500,
                            'Controls how often evaluation is carried out!')

FLAGS = tf.app.flags.FLAGS
args = FLAGS
PARAMS = args.__flags

def make_inp_fn(filename, mode, batch_size):
    
    def _input_fn(params):
        image_dataset = tf.data.TFRecordDataset(filename)
        W, H = args.crop_size[0], args.crop_size[1]

        # Create a dictionary describing the features.  
        image_feature_description = {
            'train_name': tf.FixedLenFeature([], tf.string),  
            'train_x': tf.FixedLenFeature([], tf.string),
            'train_y': tf.FixedLenFeature([], tf.string),
            'valid_name': tf.FixedLenFeature([], tf.string),  
            'valid_x': tf.FixedLenFeature([], tf.string),
            'valid_y': tf.FixedLenFeature([], tf.string)
        }
        def _parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            feature= tf.parse_single_example(example_proto, image_feature_description)
            train_x = feature['train_x']
            train_y = feature['train_y']
            valid_x = feature['valid_x']
            valid_y = feature['valid_y']
            train_name = feature['train_name']
            valid_name = feature['valid_name']

            train_x = tf.image.decode_png(train_x, channels=3)
            train_y = tf.image.decode_png(train_y, channels=3)
            valid_x = tf.image.decode_png(valid_x, channels=3)
            valid_y = tf.image.decode_png(valid_y, channels=3)
            
            train_x = tf.image.resize(train_x, (W, H))
            train_y = tf.image.resize(train_y, (W, H))
            valid_x = tf.image.resize(valid_x, (W, H))
            valid_y = tf.image.resize(valid_y, (W, H))
            
            train_y = train_y[:, :, 0]
            train_y = tf.expand_dims(train_y, axis=-1)
            
            valid_y = valid_y[:, :, 0]
            valid_y = tf.expand_dims(valid_y, axis=-1)
            
            if(args.use_bfloat):
                train_x = tf.cast(train_x, tf.bfloat16)
                train_y = tf.cast(train_y, tf.bfloat16)
                valid_x = tf.cast(valid_x, tf.bfloat16)
                valid_y = tf.cast(valid_y, tf.bfloat16)
            else:
                train_x = tf.cast(train_x, tf.float32)
                train_y = tf.cast(train_y, tf.float32)
                valid_x = tf.cast(valid_x, tf.float32)
                valid_y = tf.cast(valid_y, tf.float32)
                
            return ((train_x, valid_x), (train_y, valid_y))

        dataset = image_dataset.map(_parse_image_function)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size,  drop_remainder=True)

        return dataset
    
    return _input_fn

# evaluation hook


class GeneSaver(tf.estimator.SessionRunHook):
    def __init__(self, model, alphas_normal, alphas_reduce):
        self.model = model
        self.alphas_normal = alphas_normal
        self.alphas_reduce = alphas_reduce
    
    def begin(self):
        self.global_step = tf.train.get_or_create_global_step()
        
    def end(self, session):
        alphas_normal, alphas_reduce = session.run([self.alphas_normal, self.alphas_reduce])
        alphas_normal = softmax(alphas_normal)
        alphas_reduce = softmax(alphas_reduce)
        
        genotype = self.model.genotype(alphas_normal, alphas_reduce)        
        self.global_step = session.run(self.global_step)
        
        filename = 'final_genotype_{}'.format((self.global_step))
        tf.logging.info("Saving Genotype for step: {}".format(str(self.global_step)))
        utils.write_genotype(genotype, filename)

# model function
def one_hot_encode(x):
    b, w, h, c = x.shape
    y = tf.reshape(tf.cast(x, tf.int64), (b, w, h))
    y = tf.one_hot(y, args.num_classes, on_value=1.0, off_value=0.0)
    return y

def model_fn(features, labels, mode, params):
    criterion = tf.losses.softmax_cross_entropy
    model = Network(C=args.init_channels, net_layers=args.num_layers, criterion=criterion, num_classes=args.num_classes)
    
    eval_hooks = None
    loss = None
    train_op = None
    eval_metric_ops = None
    prediction_dict = None
    export_outputs = None
    host_call = None
    # 2. Loss function, training/eval ops
    if mode == tf.estimator.ModeKeys.TRAIN:
        (x_train, x_valid) = features
        (y_train, y_valid) = labels
        global_step = tf.train.get_global_step()
        learning_rate_min = tf.constant(args.learning_rate_min)
        num_batches_per_epoch = args.num_train_examples // args.train_batch_size
        learning_rate = tf.train.exponential_decay(
            args.learning_rate,
            global_step,
            decay_rate=args.learning_rate_decay,
            decay_steps=num_batches_per_epoch,
            staircase=False,
        )
        lr = tf.maximum(learning_rate, learning_rate_min)
        optimizer = tf.train.MomentumOptimizer(lr, args.momentum)

        if(args.use_tpu):
            optimizer = tf.tpu.CrossShardOptimizer(optimizer)


        preds = model(x_train)
        architect = Architect(model, args)
        # architect step
        tf.logging.info("Computing Architect step!")
        architect_step = architect.step(input_train=x_train,
                                        target_train=y_train,
                                        input_valid=x_valid,
                                        target_valid=y_valid,
                                        unrolled=args.unrolled,
                                        )


        w_var = model.get_thetas()
        loss = model._loss(preds, y_train)

        with tf.control_dependencies([architect_step]):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, var_list=w_var, global_step=global_step)
        
        
        if args.use_host_call:
            def host_call_fn(global_step, learning_rate, loss):
                # Outfeed supports int32 but global_step is expected to be int64.
                global_step = tf.reduce_mean(global_step)
                global_step = tf.cast(global_step, tf.int64)

                
                with (tf.contrib.summary.create_file_writer(args.model_dir).as_default()):
                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar(
                            'learning_rate', tf.reduce_mean(learning_rate),
                            step=global_step)
                        tf.contrib.summary.scalar(
                            'loss', tf.reduce_mean(loss),step=global_step)
                return tf.contrib.summary.all_summary_ops()

            global_step_t = tf.reshape(global_step, [1])
            learning_rate_t = tf.reshape(learning_rate, [1])
            loss_t = tf.reshape(loss, [1])
            host_call = (host_call_fn,
                           [global_step_t, learning_rate_t, loss_t])

    elif mode == tf.estimator.ModeKeys.EVAL:
        alphas_normal = model.arch_parameters()[0]
        alphas_reduce = model.arch_parameters()[1]
        gene_saver = GeneSaver(model, alphas_normal, alphas_reduce)
        eval_hooks = [gene_saver]
        
        (x_train, x_valid) = features
        (y_train, y_valid) = labels
        preds = model(x_valid)
        loss = model._loss(preds, y_valid)
        y = one_hot_encode(y_valid)

        metric_fn = lambda y, preds: {
            'miou': tf.metrics.mean_iou(
                    labels=y,
                    predictions=one_hot_encode(tf.expand_dims(tf.argmax(preds, axis=-1), axis=-1)),
                    num_classes=args.num_classes
                ),
            'acc': tf.metrics.accuracy(labels=y, 
                                      predictions=one_hot_encode(tf.expand_dims(tf.argmax(preds, axis=-1), axis=-1)))
        }
        eval_metric_ops = (metric_fn, [y, preds])
        
    elif mode == tf.estimator.ModeKeys.PREDICT:
        x = features
        y = labels
        preds = model(x)
        prediction_dict = {"predictions": preds}
        export_outputs = {"predict_export_outputs": tf.estimator.export.PredictOutput(outputs = preds)}
    
    # 5. Return EstimatorSpec
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode = mode,
        predictions = prediction_dict,
        loss = loss,
        train_op = train_op,
        eval_metrics = eval_metric_ops,
        export_outputs = export_outputs,
        evaluation_hooks=eval_hooks,
        host_call=host_call
    )

# Make dataset and init estimator
# Create functions to read in respective datasets

def get_train():
    return make_inp_fn(filename=tf.gfile.Glob(os.path.join(args.data, 'train-search-*-00008.tfrecords')),
                        mode=tf.estimator.ModeKeys.TRAIN,
                        batch_size=args.train_batch_size)

# Create serving input function

def serving_input_fn():
    feature_placeholders = {
        'IMAGE_LOC': tf.placeholder(tf.float32, [None])
    }

    feature_placeholders['IMAGES'] = tf.placeholder(
        tf.float32, [None, args.crop_size[0], args.crop_size[1], args.init_channels])

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

# Create custom estimator's train and evaluate function

def train_and_evaluate(estimator, input_fn):
    current_step = estimator.latest_checkpoint()
    if(not current_step):
        current_step = 0
    else:
        current_step = int(estimator.latest_checkpoint().split('-')[-1])
    tf.logging.info('Training for %d steps. Current step %d.' % (args.max_steps, current_step))

    start_timestamp = time.time()
    while current_step < args.max_steps:
        next_checkpoint = min(current_step + args.steps_per_eval, args.max_steps)
        estimator.train(input_fn=input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.' % (current_step, elapsed_time))
        tf.logging.info('Starting to evaluate.')

        eval_results = estimator.evaluate(
            input_fn=input_fn,
            steps= args.num_train_examples // args.eval_batch_size
        )

        tf.logging.info('Eval results: %s' % eval_results)

def main(argv):
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=args.tpu,
      zone=args.zone,
      project=args.project)
    
    config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=args.model_dir,
        save_checkpoints_steps=args.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=args.num_shards,
          iterations_per_loop=args.iterations_per_loop),
        tf_random_seed=int(args.seed))
    
    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=args.use_tpu,
        model_fn=model_fn,
        config=config,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        export_to_tpu=False,
        params=PARAMS
    )
    
    if args.mode == 'train_eval':
        train_and_evaluate(estimator, get_train())

    elif args.mode == 'eval':
        # TODO: change input_fn = get_valid()
        if(args.steps):
            estimator.evaluate(input_fn=get_train(), steps=args.steps)
        else:
            steps = args.num_train_examples // args.eval_batch_size
            estimator.evaluate(input_fn=get_train(), steps=steps)
    elif args.mode == 'train':
        steps = args.steps
        max_steps = None
        if steps == None:
            max_steps = args.max_steps
            
        estimator.train(input_fn=get_train(), max_steps=max_steps, steps=steps)
    else:
        raise NotImplementedError(
            'mode = {} not implemented!'.format(args.mode))


if __name__ == "__main__":
    tf.flags.mark_flag_as_required('tpu')
    tf.flags.mark_flag_as_required('zone')
    tf.flags.mark_flag_as_required('project')
    tf.flags.mark_flag_as_required('data')
    tf.flags.mark_flag_as_required('model_dir')
    
    # set random seeds
    tf.set_random_seed(int(args.seed))
    np.random.seed(int(args.seed))
                           
    tf.app.run(main=main)
