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

tf.app.flags.DEFINE_integer('num_batches_per_epoch',
                            2000,
                            'Number of examples per epoch (i.e. num_examples / batch_size)')

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
                         [512, 512],
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

tf.app.flags.DEFINE_integer('throttle_secs',
                            200,
                            'Minimum Secs to wait before performing eval.')

tf.app.flags.DEFINE_string('model_dir',
                           None,
                           'Model Events and checkpoint folder. Make sure it is a bucket.')

tf.app.flags.DEFINE_string('data',
                           None,
                           'Dataset folder for tf records, better it be bucket')

tf.app.flags.DEFINE_enum('mode',
                         'train_eval',
                         ['train_eval', 'eval', 'train'],
			 'mode=train/eval/train_eval')

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
tf.app.flags.DEFINE_integer('iterations_per_loop',
                            300,
                            'iterations_per_loop')

tf.app.flags.DEFINE_integer('start_delay_secs',
                            120,
                            'start_delay_secs')

FLAGS = tf.app.flags.FLAGS
args = FLAGS
PARAMS = args.__dict__

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
            
            
            train_x = tf.cast(train_x, tf.float32)
            train_x = tf.image.resize(train_x, (W, H))
            train_y = tf.cast(train_y, tf.float32)
            train_y = tf.image.resize(train_y, (W, H))
            valid_x = tf.cast(valid_x, tf.float32)
            valid_x = tf.image.resize(valid_x, (W, H))
            valid_y = tf.cast(valid_y, tf.float32)
            valid_y = tf.image.resize(valid_y, (W, H))
            
            train_y = train_y[:, :, 0]
            train_y = tf.expand_dims(train_y, axis=-1)
            
            valid_y = valid_y[:, :, 0]
            valid_y = tf.expand_dims(valid_y, axis=-1)
            
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
    def __init__(self, genotype):
        self.genotype = genotype

    def begin(self):
        self.global_step = tf.train.get_or_create_global_step()

    def end(self, session):
        normal_gene_op = self.genotype.normal
        reduce_gene_op = self.genotype.reduce

        self.global_step = session.run(self.global_step)
        normal_gene = session.run(normal_gene_op)
        reduce_gene = session.run(reduce_gene_op)

        genotype = Genotype(
            normal=normal_gene, normal_concat=self.genotype.normal_concat,
            reduce=reduce_gene, reduce_concat=self.genotype.reduce_concat
        )

        filename = 'final_genotype.{}'.format((self.global_step))
        tf.logging.info("Saving Genotype for step: {}".format(
            str(self.global_step)))
        utils.write_genotype(genotype, filename)

# model function

def model_fn(features, labels, mode, params):
    criterion = tf.losses.softmax_cross_entropy
    model = Network(C=args.init_channels, net_layers=args.num_layers, criterion=criterion, num_classes=args.num_classes)
    global_step = tf.train.get_global_step()
    learning_rate_min = tf.constant(args.learning_rate_min)
    
    learning_rate = tf.train.exponential_decay(
        args.learning_rate,
        global_step,
        decay_rate=args.learning_rate_decay,
        decay_steps=args.num_batches_per_epoch,
        staircase=True,
    )
    
    lr = tf.maximum(learning_rate, learning_rate_min)
#     tf.summary.scalar('learning_rate', lr)
    
    optimizer = tf.train.MomentumOptimizer(lr, args.momentum)
    criterion = tf.losses.softmax_cross_entropy
    
    if(args.use_tpu):
        optimizer = tf.tpu.CrossShardOptimizer(optimizer)
        
    eval_hooks = None
    
    loss = None
    train_op = None
    eval_metric_ops = None
    prediction_dict = None
    export_outputs = None
    # 2. Loss function, training/eval ops
    if mode == tf.estimator.ModeKeys.TRAIN:
        (x_train, x_valid) = features
        (y_train, y_valid) = labels

        preds = model(x_train)
        architect = Architect(model, args)
        # architect step
        architect_step = architect.step(input_train=x_train,
                                        target_train=y_train,
                                        input_valid=x_valid,
                                        target_valid=y_valid,
                                        unrolled=args.unrolled,
                                        )


        w_var = model.get_thetas()
        loss = model._loss(preds, y_train)
        grads = tf.gradients(loss, w_var)
        clipped_gradients, norm = tf.clip_by_global_norm(grads, args.grad_clip)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, w_var), global_step=tf.train.get_global_step())
        
        b, w, h, c = y_train.shape
        y = tf.reshape(tf.cast(y_train, tf.int64), (b, w, h))
        y = tf.one_hot(y, args.num_classes, on_value=1.0, off_value=0.0)
        
        metric_fn = lambda y, preds: {
            'miou': tf.metrics.mean_iou(
                    labels=y,
                    predictions=tf.round(preds),
                    num_classes=args.num_classes
                ),
        
            'acc': tf.metrics.accuracy(labels=y, 
                                      predictions=tf.round(preds))
        }
        eval_metric_ops = (metric_fn, [y, preds])
        
        host_call = None
        if(args.use_host_call):
            def host_call_fn(global_step, learning_rate):
                # Outfeed supports int32 but global_step is expected to be int64.
                global_step = tf.reduce_mean(global_step)
                global_step = tf.cast(global_step, tf.int64)

                with (tf.contrib.summary.create_file_writer(args.model_dir).as_default()):
                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar(
                            'learning_rate', tf.reduce_mean(learning_rate),
                            step=global_step)
                return tf.contrib.summary.all_summary_ops()
            
            global_step_t = tf.reshape(global_step, [1])
            learning_rate_t = tf.reshape(learning_rate, [1])
            host_call = (host_call_fn,
                           [global_step_t, learning_rate_t])

    elif mode == tf.estimator.ModeKeys.EVAL:
        genotype = model.genotype()
        gene_saver = GeneSaver(genotype)
        eval_hooks = [gene_saver]
        
        (x_train, x_valid) = features
        (y_train, y_valid) = labels
        preds = model(x_valid)
        loss = model._loss(preds, y_valid)
        
        b, w, h, c = y_valid.shape
        y = tf.reshape(tf.cast(y_valid, tf.int64), (b, w, h))
        y = tf.one_hot(y, args.num_classes, on_value=1.0, off_value=0.0)
        
        metric_fn = lambda y, preds: {
            'miou': tf.metrics.mean_iou(
                    labels=y,
                    predictions=tf.round(preds),
                    num_classes=args.num_classes
                ),
        
            'acc': tf.metrics.accuracy(labels=y, 
                                      predictions=tf.round(preds))
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


def train_and_evaluate(output_dir, estimator):
    train_spec = tf.estimator.TrainSpec(input_fn=get_train(),
                                        max_steps=args.max_steps)
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=get_train(),
                                      steps=None, start_delay_secs=args.start_delay_secs, throttle_secs=args.throttle_secs)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

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
          iterations_per_loop=args.iterations_per_loop,
          num_shards=args.num_shards))
    
    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=config,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        params=PARAMS
    )
    
    if args.mode == 'train_eval':
        train_and_evaluate(args.model_dir, estimator)

    elif args.mode == 'eval':
        # TODO: change input_fn = get_valid()
        estimator.evaluate(input_fn=get_train(), steps=args.steps)
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
