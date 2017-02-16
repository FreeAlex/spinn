"""

Multiple Models:

1. Specify JSON files that contain all necessary information for each model.
2. Keep the args from the JSON file. Load the ckpt specified in JSON file. Create
    model based off the args.

Missing Features:

1. Loss
2. Eval Report
3. Eval/Train Mode
4. Set GPU
5. Set Debug
6. Transition Accuracy

"""

import os
import sys
import json
import pprint
import time
from collections import deque

import gflags
import numpy as np

from spinn import afs_safe_logger
from spinn import util
from spinn.data.arithmetic import load_simple_data
from spinn.data.boolean import load_boolean_data
from spinn.data.sst import load_sst_data
from spinn.data.snli import load_snli_data
from spinn.util.data import SimpleProgressBar, is_sequential_only, truncate, get_checkpoint_path
from spinn.util.blocks import the_gpu, to_gpu, l2_cost, flatten, debug_gradient
from spinn.util.misc import Accumulator, time_per_token, MetricsLogger, EvalReporter

import spinn.rl_spinn
import spinn.fat_stack
import spinn.plain_rnn
import spinn.cbow

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


FLAGS = gflags.FLAGS


class EnsembleArgs(object):
    pass


class EnsembleModel(object):
    pass


class EnsembleTrainer(object):
    def __init__(self, data_manager, initial_embeddings):
        super(EnsembleTrainer, self).__init__()
        self.initial_embeddings = initial_embeddings
        self.data_manager = data_manager
        self.vocab_size = initial_embeddings.shape[0]
        self.num_classes = len(data_manager.LABEL_MAP)
        self.use_sentence_pair = data_manager.SENTENCE_PAIR_DATA
        self.models = []

    def load_args(self, json_file):
        with open(json_file) as f:
            args_dict = json.loads(f.read())
        args = EnsembleArgs()
        for k, v in args_dict.iteritems():
            v = str(v) if type(v) == unicode else v
            setattr(args, k, v)
        return args

    def get_classes(self, args):
        if args.model_type == "CBOW":
            model_module = spinn.cbow
        elif args.model_type == "RNN":
            model_module = spinn.plain_rnn
        elif args.model_type == "SPINN":
            model_module = spinn.fat_stack
        elif args.model_type == "RLSPINN":
            model_module = spinn.rl_spinn
        else:
            raise Exception("Requested unimplemented model type %s" % FLAGS.model_type)

        if self.use_sentence_pair:
            trainer_cls = model_module.SentencePairTrainer
            model_cls = model_module.SentencePairModel
        else:
            trainer_cls = model_module.SentenceTrainer
            model_cls = model_module.SentenceModel
        return trainer_cls, model_cls

    def build_model(self, args, model_cls):

        initial_embeddings = self.initial_embeddings
        vocab_size = self.vocab_size
        use_sentence_pair = self.use_sentence_pair
        num_classes = self.num_classes

        model = model_cls(model_dim=args.model_dim,
            word_embedding_dim=args.word_embedding_dim,
            vocab_size=vocab_size,
            initial_embeddings=initial_embeddings,
            num_classes=num_classes,
            mlp_dim=args.mlp_dim,
            embedding_keep_rate=args.embedding_keep_rate,
            classifier_keep_rate=args.semantic_classifier_keep_rate,
            tracking_lstm_hidden_dim=args.tracking_lstm_hidden_dim,
            transition_weight=args.transition_weight,
            use_encode=args.use_encode,
            encode_reverse=args.encode_reverse,
            encode_bidirectional=args.encode_bidirectional,
            encode_num_layers=args.encode_num_layers,
            use_sentence_pair=use_sentence_pair,
            use_skips=args.use_skips,
            lateral_tracking=args.lateral_tracking,
            use_tracking_in_composition=args.use_tracking_in_composition,
            use_difference_feature=args.use_difference_feature,
            use_product_feature=args.use_product_feature,
            num_mlp_layers=args.num_mlp_layers,
            mlp_bn=args.mlp_bn,
            rl_mu=args.rl_mu,
            rl_baseline=args.rl_baseline,
            rl_reward=args.rl_reward,
            rl_weight=args.rl_weight,
        )

        return model

    def build_optimizer(self, args, model):
        # Build optimizer.
        if args.optimizer_type == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        elif args.optimizer_type == "RMSProp":
            optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, eps=1e-08)
        else:
            raise NotImplementedError
        return optimizer

    def build_trainer(self, args, model, optimizer, trainer_cls):
        return trainer_cls(model, optimizer)

    def _add_model(self, name, args, model, optimizer, trainer, step, best_dev_error):
        ensemble_model = EnsembleModel()
        ensemble_model.name = name if name is not None else args.experiment_name
        ensemble_model.args = args
        ensemble_model.model = model
        ensemble_model.optimizer = optimizer
        ensemble_model.step = step
        ensemble_model.trainer = trainer
        ensemble_model.best_dev_error = best_dev_error

        self.models.append(ensemble_model)

    def add_model(self, json_file, name=None):
        args = self.load_args(json_file)

        trainer_cls, model_cls = self.get_classes(args)

        model = self.build_model(args, model_cls)
        optimizer = self.build_optimizer(args, model)
        trainer = self.build_trainer(args, model, optimizer, trainer_cls)

        standard_ckpt_path = get_checkpoint_path(args.ckpt_path, args.experiment_name)
        step, best_dev_error = trainer.load(standard_ckpt_path)

        self._add_model(name, args, model, optimizer, trainer, step, best_dev_error)

    def __call__(self, x_batch, transitions_batch, y_batch=None,
                 use_internal_parser=False, validate_transitions=True):
        outputs = []
        for ensemble_model in self.models:
            model = ensemble_model.model
            args = ensemble_model.args

            outp = model(x_batch, transitions_batch, y_batch,
                use_internal_parser=args.use_internal_parser,
                validate_transitions=args.validate_transitions)

            outputs.append(outp)

        return outputs


def evaluate(model, eval_set, logger, metrics_logger, step, sequential_only, vocabulary=None):
    filename, dataset = eval_set

    # Evaluate
    class_correct = 0
    class_total = 0
    total_batches = len(dataset)
    progress_bar = SimpleProgressBar(msg="Run Eval", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)
    total_tokens = 0
    start = time.time()

    metrics = []

    for i, (eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch, eval_ids) in enumerate(dataset):
        if FLAGS.truncate_eval_batch:
            eval_X_batch, eval_transitions_batch = truncate(
                eval_X_batch, eval_transitions_batch, eval_num_transitions_batch, sequential_only, FLAGS.use_left_padding)

        # Run model.
        output = model(eval_X_batch, eval_transitions_batch, eval_y_batch,
            use_internal_parser=FLAGS.use_internal_parser,
            validate_transitions=FLAGS.validate_transitions)

        output = torch.cat([outp.unsqueeze(0) for outp in output], 0)

        normalized_output = torch.cat([F.softmax(oo).unsqueeze(0) for oo in output], 0)
        if FLAGS.ensemble_type == "mean":
            modified = normalized_output.mean(0).squeeze(0)
        elif FLAGS.ensemble_type == "max":
            modified = normalized_output.max(0)[0].squeeze(0)
        else:
            raise NotImplementedError

        # Calculate class accuracy.
        target = torch.from_numpy(eval_y_batch).long()
        pred = modified.data.max(1)[1].cpu() # get the index of the max log-probability
        class_correct += pred.eq(target).sum()
        class_total += target.size(0)

        metrics.append([target, normalized_output])

        # Optionally calculate transition loss/acc.
        transition_loss = model.transition_loss if hasattr(model, 'transition_loss') else None

        # Update Aggregate Accuracies
        total_tokens += eval_num_transitions_batch.ravel().sum()

        # Print Progress
        progress_bar.step(i+1, total=total_batches)
    progress_bar.finish()

    end = time.time()
    total_time = end - start

    # Get time per token.
    time_metric = time_per_token([total_tokens], [total_time])

    # Get class accuracy.
    eval_class_acc = class_correct / float(class_total)

    # TODO: Transition Accuracy
    eval_trans_acc = 0.0

    logger.Log("Step: %i Eval acc: %f  %f %s Time: %5f" %
              (step, eval_class_acc, eval_trans_acc, filename, time_metric))

    metrics_logger.Log('eval_class_acc', eval_class_acc, step)
    metrics_logger.Log('eval_trans_acc', eval_trans_acc, step)

    if FLAGS.write_ensemble_report:
        ensemble_report_path = os.path.join(FLAGS.log_path, FLAGS.experiment_name + ".ensemble_report")
        all_targets, all_outp = zip(*metrics)
        all_targets = torch.cat(all_targets, 0)
        all_outp = torch.cat(all_outp, 1)
        with open(ensemble_report_path, "w") as f:
            for i in range(all_targets.size(0)):
                row = "{target},{dists}\n"
                target = all_targets[i]
                dists = all_outp[:, 0].contiguous().view(-1).data.tolist()
                f.write(row.format(target=target, dists=",".join(str(d) for d in dists)))

    return eval_class_acc


def run(only_forward=False):
    logger = afs_safe_logger.Logger(os.path.join(FLAGS.log_path, FLAGS.experiment_name) + ".log")

    # Select data format.
    if FLAGS.data_type == "bl":
        data_manager = load_boolean_data
    elif FLAGS.data_type == "sst":
        data_manager = load_sst_data
    elif FLAGS.data_type == "snli":
        data_manager = load_snli_data
    elif FLAGS.data_type == "arithmetic":
        data_manager = load_simple_data
    else:
        logger.Log("Bad data type.")
        return

    pp = pprint.PrettyPrinter(indent=4)
    logger.Log("Flag values:\n" + pp.pformat(FLAGS.FlagValuesDict()))

    # Make Metrics Logger.
    metrics_path = "{}/{}".format(FLAGS.metrics_path, FLAGS.experiment_name)
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    metrics_logger = MetricsLogger(metrics_path)
    M = Accumulator(maxlen=FLAGS.deque_length)

    # Load the data.
    raw_training_data, vocabulary = data_manager.load_data(
        FLAGS.training_data_path, FLAGS.lowercase)

    # Load the eval data.
    raw_eval_sets = []
    if FLAGS.eval_data_path:
        for eval_filename in FLAGS.eval_data_path.split(":"):
            raw_eval_data, _ = data_manager.load_data(eval_filename, FLAGS.lowercase)
            raw_eval_sets.append((eval_filename, raw_eval_data))

    # Prepare the vocabulary.
    if not vocabulary:
        logger.Log("In open vocabulary mode. Using loaded embeddings without fine-tuning.")
        train_embeddings = False
        vocabulary = util.BuildVocabulary(
            raw_training_data, raw_eval_sets, FLAGS.embedding_data_path, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    else:
        logger.Log("In fixed vocabulary mode. Training embeddings.")
        train_embeddings = True

    # Load pretrained embeddings.
    if FLAGS.embedding_data_path:
        logger.Log("Loading vocabulary with " + str(len(vocabulary))
                   + " words from " + FLAGS.embedding_data_path)
        initial_embeddings = util.LoadEmbeddingsFromText(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path)
    else:
        initial_embeddings = None

    sequential_only = False

    # Preprocess eval sets.
    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        eval_data = util.PreprocessDataset(
            raw_eval_set, vocabulary,
            FLAGS.eval_seq_length if FLAGS.eval_seq_length is not None else FLAGS.seq_length,
            data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=sequential_only,
            use_left_padding=FLAGS.use_left_padding)
        eval_it = util.MakeEvalIterator(eval_data,
            FLAGS.batch_size, FLAGS.eval_data_limit, bucket_eval=FLAGS.bucket_eval,
            shuffle=FLAGS.shuffle_eval, rseed=FLAGS.shuffle_eval_seed)
        eval_iterators.append((filename, eval_it))

    step = -1
    model = EnsembleTrainer(data_manager, initial_embeddings)
    ensemble_paths = FLAGS.ensemble_path.split(',')
    for ep in ensemble_paths:
        model.add_model(ep)

    # Ensemble only supports eval right now.
    for index, eval_set in enumerate(eval_iterators):
        acc = evaluate(model, eval_set, logger, metrics_logger, step, sequential_only, vocabulary)


if __name__ == '__main__':
    # Debug settings.
    gflags.DEFINE_bool("debug", False, "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_bool("show_progress_bar", True, "Turn this off when running experiments on HPC.")
    gflags.DEFINE_string("branch_name", "", "")
    gflags.DEFINE_integer("deque_length", None, "Max trailing examples to use for statistics.")
    gflags.DEFINE_string("sha", "", "")
    gflags.DEFINE_string("experiment_name", "", "")

    # Data types.
    gflags.DEFINE_enum("data_type", "bl", ["bl", "sst", "snli", "arithmetic"],
        "Which data handler and classifier to use.")

    # Where to store checkpoints
    gflags.DEFINE_string("ckpt_path", ".", "Where to save/load checkpoints. Can be either "
        "a filename or a directory. In the latter case, the experiment name serves as the "
        "base for the filename.")
    gflags.DEFINE_string("metrics_path", ".", "A directory in which to write logs.")
    gflags.DEFINE_string("log_path", ".", "A directory in which to write logs.")
    gflags.DEFINE_integer("ckpt_step", 1000, "Steps to run before considering saving checkpoint.")
    gflags.DEFINE_boolean("load_best", False, "If True, attempt to load 'best' checkpoint.")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "Can contain multiple file paths, separated "
        "using ':' tokens. The first file should be the dev set, and is used for determining "
        "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_integer("seq_length", 30, "")
    gflags.DEFINE_integer("eval_seq_length", None, "")
    gflags.DEFINE_boolean("truncate_eval_batch", True, "Shorten batches to max transition length.")
    gflags.DEFINE_boolean("truncate_train_batch", True, "Shorten batches to max transition length.")
    gflags.DEFINE_boolean("smart_batching", True, "Organize batches using sequence length.")
    gflags.DEFINE_boolean("use_peano", True, "A mind-blowing sorting key.")
    gflags.DEFINE_integer("eval_data_limit", -1, "Truncate evaluation set. -1 indicates no truncation.")
    gflags.DEFINE_boolean("bucket_eval", True, "Bucket evaluation data for speed improvement.")
    gflags.DEFINE_boolean("shuffle_eval", False, "Shuffle evaluation data.")
    gflags.DEFINE_integer("shuffle_eval_seed", 123, "Seed shuffling of eval data.")
    gflags.DEFINE_string("embedding_data_path", None,
        "If set, load GloVe-formatted embeddings from here.")

    # Data preprocessing settings.
    gflags.DEFINE_boolean("use_skips", False, "Pad transitions with SKIP actions.")
    gflags.DEFINE_boolean("use_left_padding", True, "Pad transitions only on the LHS.")

    # Ensemble settings.
    gflags.DEFINE_string("ensemble_path", None, "List of comma-seperated files that hold"
        "JSON arguments for saved models.")
    gflags.DEFINE_enum("ensemble_type", "mean", ["mean", "max"], "")
    gflags.DEFINE_boolean("write_ensemble_report", True, "")

    # Model architecture settings.
    gflags.DEFINE_enum("model_type", "RNN", ["CBOW", "RNN", "SPINN", "RLSPINN"], "")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")
    gflags.DEFINE_boolean("lowercase", False, "When True, ignore case.")
    gflags.DEFINE_boolean("use_internal_parser", False, "Use predicted parse.")
    gflags.DEFINE_boolean("validate_transitions", True,
        "Constrain predicted transitions to ones that give a valid parse tree.")
    gflags.DEFINE_float("embedding_keep_rate", 0.9,
        "Used for dropout on transformed embeddings.")
    gflags.DEFINE_boolean("force_transition_loss", False, "")
    gflags.DEFINE_boolean("use_l2_cost", True, "")
    gflags.DEFINE_boolean("use_difference_feature", True, "")
    gflags.DEFINE_boolean("use_product_feature", True, "")

    # Tracker settings.
    gflags.DEFINE_integer("tracking_lstm_hidden_dim", None, "Set to none to avoid using tracker.")
    gflags.DEFINE_float("transition_weight", None, "Set to none to avoid predicting transitions.")
    gflags.DEFINE_boolean("lateral_tracking", True,
        "Use previous tracker state as input for new state.")
    gflags.DEFINE_boolean("use_tracking_in_composition", True,
        "Use tracking lstm output as input for the reduce function.")

    # Encode settings.
    gflags.DEFINE_boolean("use_encode", False, "Encode embeddings with sequential network.")
    gflags.DEFINE_boolean("encode_reverse", False, "Encode in reverse order.")
    gflags.DEFINE_boolean("encode_bidirectional", False, "Encode in both directions.")
    gflags.DEFINE_integer("encode_num_layers", 1, "RNN layers in encoding net.")

    # RL settings.
    gflags.DEFINE_float("rl_mu", 0.1, "Use in exponential moving average baseline.")
    gflags.DEFINE_enum("rl_baseline", "ema", ["ema", "greedy", "policy"],
        "Different configurations to approximate reward function.")
    gflags.DEFINE_enum("rl_reward", "standard", ["standard", "xent"],
        "Different reward functions to use.")
    gflags.DEFINE_float("rl_weight", 1.0, "Hyperparam for REINFORCE loss.")

    # MLP settings.
    gflags.DEFINE_integer("mlp_dim", 1024, "Dimension of intermediate MLP layers.")
    gflags.DEFINE_integer("num_mlp_layers", 2, "Number of MLP layers.")
    gflags.DEFINE_boolean("mlp_bn", True, "When True, batch normalization is used between MLP layers.")
    gflags.DEFINE_float("semantic_classifier_keep_rate", 0.9,
        "Used for dropout in the semantic task classifier.")

    # Optimization settings.
    gflags.DEFINE_enum("optimizer_type", "Adam", ["Adam", "RMSprop"], "")
    gflags.DEFINE_integer("training_steps", 500000, "Stop training after this point.")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in optimizer.")
    gflags.DEFINE_float("learning_rate_decay_per_10k_steps", 0.75, "Used in optimizer.")
    gflags.DEFINE_boolean("actively_decay_learning_rate", True, "Used in optimizer.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float("init_range", 0.005, "Mainly used for softmax parameters. Range for uniform random init.")

    # Display settings.
    gflags.DEFINE_integer("statistics_interval_steps", 100, "Print training set results at this interval.")
    gflags.DEFINE_integer("metrics_interval_steps", 10, "Evaluate at this interval.")
    gflags.DEFINE_integer("eval_interval_steps", 100, "Evaluate at this interval.")
    gflags.DEFINE_integer("ckpt_interval_steps", 5000, "Update the checkpoint on disk at this interval.")
    gflags.DEFINE_boolean("ckpt_on_best_dev_error", True, "If error on the first eval set (the dev set) is "
        "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")

    # Evaluation settings
    gflags.DEFINE_boolean("expanded_eval_only_mode", False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
        "transitions. The inferred parses are written to the supplied file(s) along with example-"
        "by-example accuracy information. Requirements: Must specify checkpoint path.")
    gflags.DEFINE_boolean("write_eval_report", False, "")

    # Parse command line flags.
    FLAGS(sys.argv)

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}-{}".format(
            FLAGS.data_type,
            FLAGS.model_type,
            timestamp,
            )

    if not FLAGS.branch_name:
        FLAGS.branch_name = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.sha:
        FLAGS.sha = os.popen('git rev-parse HEAD').read().strip()

    run(only_forward=FLAGS.expanded_eval_only_mode)
