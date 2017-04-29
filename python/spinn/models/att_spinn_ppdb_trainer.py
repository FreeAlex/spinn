"""From the project root directory (containing data files), this can be run with:

Boolean logic evaluation:
python -m spinn.models.fat_classifier --training_data_path ../bl-data/pbl_train.tsv \
       --eval_data_path ../bl-data/pbl_dev.tsv

SST sentiment (Demo only, model needs a full GloVe embeddings file to do well):
python -m spinn.models.fat_classifier --data_type sst --training_data_path sst-data/train.txt \
       --eval_data_path sst-data/dev.txt --embedding_data_path spinn/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

SNLI entailment (Demo only, model needs a full GloVe embeddings file to do well):
python -m spinn.models.fat_classifier --data_type snli --training_data_path snli_1.0/snli_1.0_dev.jsonl \
       --eval_data_path snli_1.0/snli_1.0_dev.jsonl --embedding_data_path spinn/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

Note: If you get an error starting with "TypeError: ('Wrong number of dimensions..." during development,
    there may already be a saved checkpoint in ckpt_path that matches the name of the model you're developing.
    Move or delete it as appropriate.
"""

import os
import pprint
import sys
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
from spinn.util.data import SimpleProgressBar
from spinn.util.blocks import the_gpu, to_gpu, l2_cost, flatten, debug_gradient
from spinn.util.misc import Accumulator, time_per_token, MetricsLogger, EvalReporter
from spinn.data.ppdb import load as load_ppdb
import spinn.att_spinn_multi_mlp
from itertools import izip
import resource

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import logging

FLAGS = gflags.FLAGS

def get_mem_use():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    mem = usage[2]
    measures = ['B', 'KB', 'MB', 'GB', 'TB']
    for m in measures:
        if mem > 1024:
            mem /= 1024.0
        else:
            return '{:.2f}{}'.format(mem, m)
    ValueError("Memory usage too big, bug?")

def define_flags():
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
    gflags.DEFINE_integer("seq_length", 60, "Sentence are paded with zeros when it's transitions are over this size, #transitions = 2*#tokens -1")
    gflags.DEFINE_integer("eval_seq_length", None, "")
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

    # Model architecture settings.
    gflags.DEFINE_enum("model_type", "ATT_SPINN_PPDB", ["ATT_SPINN_PPDB"], "")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")
    gflags.DEFINE_boolean("lowercase", False, "When True, ignore case.")
    gflags.DEFINE_boolean("use_internal_parser", False, "Use predicted parse.")
    gflags.DEFINE_boolean("validate_transitions", True,
                          "Constrain predicted transitions to ones that give a valid parse tree.")
    gflags.DEFINE_float("embedding_keep_rate", 0.9,
                        "Used for dropout on transformed embeddings and in the encoder RNN.")
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
    gflags.DEFINE_enum("encode_style", None, ["LSTM", "CNN", "QRNN"], "Encode embeddings with sequential context.")
    gflags.DEFINE_boolean("encode_reverse", False, "Encode in reverse order.")
    gflags.DEFINE_boolean("encode_bidirectional", False, "Encode in both directions.")
    gflags.DEFINE_integer("encode_num_layers", 1, "RNN layers in encoding net.")

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
    gflags.DEFINE_integer("statistics_interval_steps", 100, "when to plot and keep statistic")
    gflags.DEFINE_integer("eval_interval_steps", 1000, "When to do an evaluation and keep checkpoint")
    gflags.DEFINE_boolean("ckpt_on_best_dev_error", True, "If error on the first eval set (the dev set) is "
                                                          "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")

    # Attention model settings
    gflags.DEFINE_boolean("using_diff_in_mlstm", False,
                          "use (ak - hk) as a feature, ak is attention vector, hk is the vector of hypothesis in step k")
    gflags.DEFINE_boolean("using_prod_in_mlstm", False,
                          "use (ak * hk) as a feature, ak is attention vector, hk is the vector of hypothesis in step k")
    gflags.DEFINE_boolean("using_null_in_attention", False,
                          "use null vector in premise stack so that some weights can be assigned")
    gflags.DEFINE_boolean("saving_eval_attention_matrix", False, "Whether to save the attention matrix when evaluation")
    gflags.DEFINE_boolean("using_only_mlstm_feature", False,
                          "Whether to use only matching lstm as the feature input in MLP")

    # PPDB data settings
    gflags.DEFINE_string("ppdb_path", None, "ppdb data path")
    gflags.DEFINE_integer("ppdb_limit", None, "number of samples used in training, unlimited if none")
    gflags.DEFINE_float("ppdb_loss_weight", 1.0, "weight of ppdb loss")

    # Evaluation settings
    gflags.DEFINE_boolean("expanded_eval_only_mode", False,
                          "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
                          "transitions. The inferred parses are written to the supplied file(s) along with example-"
                          "by-example accuracy information. Requirements: Must specify checkpoint path.")
    gflags.DEFINE_boolean("write_eval_report", False, "")


def sequential_only():
    return FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW"


def truncate(X_batch, transitions_batch, num_transitions_batch):
    # Truncate each batch to max length within the batch.
    X_batch_is_left_padded = (not FLAGS.use_left_padding or sequential_only())
    transitions_batch_is_left_padded = FLAGS.use_left_padding
    max_transitions = np.max(num_transitions_batch)
    seq_length = X_batch.shape[1]

    if X_batch_is_left_padded:
        X_batch = X_batch[:, seq_length - max_transitions:]
    else:
        X_batch = X_batch[:, :max_transitions]

    if transitions_batch_is_left_padded:
        transitions_batch = transitions_batch[:, seq_length - max_transitions:]
    else:
        transitions_batch = transitions_batch[:, :max_transitions]

    return X_batch, transitions_batch


def evaluate(model, eval_set, logger, metrics_logger, step, vocabulary=None):
    filename, dataset = eval_set

    reporter = EvalReporter()

    # Evaluate
    class_correct = 0
    class_total = 0
    total_batches = len(dataset)
    progress_bar = SimpleProgressBar(msg="Run Eval", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)
    total_tokens = 0
    start = time.time()

    model.eval()

    transition_preds = []
    transition_targets = []

    for i, (eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch, eval_ids) in enumerate(dataset):

        if FLAGS.saving_eval_attention_matrix:
            model.set_recording_attention_weight_matrix(True)

        # Run model.
        output = model(eval_X_batch, eval_transitions_batch,
                       True,        # main dataset
                       eval_y_batch,
            use_internal_parser=FLAGS.use_internal_parser,
            validate_transitions=FLAGS.validate_transitions,)

        if FLAGS.saving_eval_attention_matrix:
            # WARNING: only attention SPINN model have attention matrix
            attention_matrix = model.get_attention_matrix_from_last_forward()
            with open(os.path.join(FLAGS.metrics_path, FLAGS.experiment_name, 'attention-matrix-{}.txt'.format(step)), 'a') as txtfile:
                for eval_id, attmat in izip(eval_ids, attention_matrix):
                    txtfile.write('{}\n'.format(eval_id))
                    txtfile.write('{},{}\n'.format(len(attmat), len(attmat[0])))
                    for row in attmat:
                        txtfile.write(','.join(['{:.1f}'.format(x*100.0) for x in row]))
                        txtfile.write('\n')
            model.set_recording_attention_weight_matrix(False) # reset it after run



        # Normalize output.
        logits = F.log_softmax(output)

        # Calculate class accuracy.
        target = torch.from_numpy(eval_y_batch).long()
        pred = logits.data.max(1)[1].cpu() # get the index of the max log-probability
        class_correct += pred.eq(target).sum()
        class_total += target.size(0)

        # Optionally calculate transition loss/acc.
        transition_loss = model.transition_loss if hasattr(model, 'transition_loss') else None

        # Update Aggregate Accuracies
        total_tokens += eval_num_transitions_batch.ravel().sum()

        # Accumulate stats for transition accuracy.
        if transition_loss is not None:
            transition_preds.append([m["t_preds"] for m in model.spinn.memories])
            transition_targets.append([m["t_given"] for m in model.spinn.memories])

        if FLAGS.write_eval_report:
            reporter_args = [pred, target, eval_ids, output.data.cpu().numpy()]
            if hasattr(model, 'transition_loss'):
                transition_preds_per_example = model.spinn.get_transition_preds_per_example()
                if model.use_sentence_pair:
                    batch_size = pred.size(0)
                    sent1_preds = transition_preds_per_example[:batch_size]
                    sent2_preds = transition_preds_per_example[batch_size:]
                    reporter_args.append(sent1_preds)
                    reporter_args.append(sent2_preds)
                else:
                    reporter_args.append(transition_preds_per_example)
            reporter.save_batch(*reporter_args)

        # Print Progress
        progress_bar.step(i+1, total=total_batches)
    progress_bar.finish()

    end = time.time()
    total_time = end - start

    # Get time per token.
    time_metric = time_per_token([total_tokens], [total_time])

    # Get class accuracy.
    eval_class_acc = class_correct / float(class_total)

    # Get transition accuracy if applicable.
    if len(transition_preds) > 0:
        all_preds = np.array(flatten(transition_preds))
        all_truth = np.array(flatten(transition_targets))
        eval_trans_acc = (all_preds == all_truth).sum() / float(all_truth.shape[0])
    else:
        eval_trans_acc = 0.0

    logger.Log("Step: %i Eval acc: %f  %f %s Time: %5f" %
              (step, eval_class_acc, eval_trans_acc, filename, time_metric))

    metrics_logger.Log('eval_class_acc', eval_class_acc, step)
    metrics_logger.Log('eval_trans_acc', eval_trans_acc, step)

    if FLAGS.write_eval_report:
        eval_report_path = os.path.join(FLAGS.log_path, FLAGS.experiment_name + ".report")
        reporter.write_report(eval_report_path)

    return eval_class_acc


def get_checkpoint_path(ckpt_path, experiment_name, suffix=".ckpt", best=False):
    # Set checkpoint path.
    if ckpt_path.endswith(suffix):
        checkpoint_path = ckpt_path
    else:
        checkpoint_path = os.path.join(ckpt_path, experiment_name + suffix)
    if best:
        checkpoint_path += "_best"
    return checkpoint_path


def run(only_forward=False):
    logger = afs_safe_logger.Logger(os.path.join(FLAGS.log_path, FLAGS.experiment_name) + ".log")

    print 'memory usage, at begin: {}'.format(get_mem_use())

    # Select data format.
    if FLAGS.data_type == "snli":
        data_manager = load_snli_data
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

    # Load the data.
    raw_training_data, vocabulary = data_manager.load_data(
        FLAGS.training_data_path, FLAGS.lowercase)
    print 'memory usage, after loading main training data: {}'.format(get_mem_use())

    # Load the eval data.
    raw_eval_sets = []
    if FLAGS.eval_data_path:
        for eval_filename in FLAGS.eval_data_path.split(":"):
            raw_eval_data, _ = data_manager.load_data(eval_filename, FLAGS.lowercase)
            raw_eval_sets.append((eval_filename, raw_eval_data))
    print 'memory usage, after loading main evaluation data: {}'.format(get_mem_use())

    # Load the ppdb data.
    logger.Log("Loading ppdb data.")
    raw_ppdb_data, _ = load_ppdb.load_data(FLAGS.ppdb_path, FLAGS.lowercase, limit=FLAGS.ppdb_limit)
    logger.Log('PPDB data size: {}'.format(len(raw_ppdb_data)))
    print 'memory usage, after loading ppdb data: {}'.format(get_mem_use())

    # Prepare the vocabulary.
    if not vocabulary:
        logger.Log("In open vocabulary mode. Using loaded embeddings without fine-tuning.")
        vocabulary = util.BuildVocabulary(
            raw_training_data + raw_ppdb_data, raw_eval_sets, FLAGS.embedding_data_path, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    else:
        logger.Log("In fixed vocabulary mode. Training embeddings.")
    print 'memory usage, after building vocabulary: {}'.format(get_mem_use())

    # Load pretrained embeddings.
    if FLAGS.embedding_data_path:
        logger.Log("Loading vocabulary with " + str(len(vocabulary))
                   + " words from " + FLAGS.embedding_data_path)
        initial_embeddings = util.LoadEmbeddingsFromText(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path)
    else:
        initial_embeddings = None
    print 'memory usage, after loading embedding: {}'.format(get_mem_use())


    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    logger.Log("Preprocessing training data.")
    training_data = util.PreprocessDataset(
        raw_training_data, vocabulary, FLAGS.seq_length, data_manager, eval_mode=False, logger=logger,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
        for_rnn=sequential_only(),
        use_left_padding=FLAGS.use_left_padding)
    training_data_iter = util.MakeTrainingIterator(
        training_data, FLAGS.batch_size, FLAGS.smart_batching, FLAGS.use_peano,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    del raw_training_data   # release raw training data
    print 'memory usage, after preprocessing main training data: {}'.format(get_mem_use())

    # Preprocess eval sets.
    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        eval_data = util.PreprocessDataset(
            raw_eval_set, vocabulary,
            FLAGS.eval_seq_length if FLAGS.eval_seq_length is not None else FLAGS.seq_length,
            data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=sequential_only(),
            use_left_padding=FLAGS.use_left_padding)
        eval_it = util.MakeEvalIterator(eval_data,
            FLAGS.batch_size, FLAGS.eval_data_limit, bucket_eval=FLAGS.bucket_eval,
            shuffle=FLAGS.shuffle_eval, rseed=FLAGS.shuffle_eval_seed)
        eval_iterators.append((filename, eval_it))
    del raw_eval_sets   # release memory resource
    print 'memory usage, after preprocessing main eval data: {}'.format(get_mem_use())
    # Preprocess ppdb data
    logger.Log("Preprocessing ppdb data.")
    ppdb_data = util.PreprocessDataset(raw_ppdb_data, vocabulary, FLAGS.seq_length, load_ppdb, eval_mode=False, logger=logger,
                                       sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
                                       for_rnn=sequential_only(),
                                       use_left_padding=FLAGS.use_left_padding)
    del raw_ppdb_data   # release memeory resource
    ppdb_training_data_iter = util.MakeTrainingIterator(ppdb_data, FLAGS.batch_size, FLAGS.smart_batching, FLAGS.use_peano,
        sentence_pair_data=load_ppdb.SENTENCE_PAIR_DATA)
    logger.Log("Preprocessing ppdb data complete.")
    print 'memory usage, after preprocessing ppdb data: {}'.format(get_mem_use())



    # Choose model.
    model_specific_params = {}
    logger.Log("Building model.")
    if FLAGS.model_type == "ATT_SPINN_PPDB":
        model_module = spinn.att_spinn_multi_mlp
        model_specific_params['using_diff_in_mlstm'] = FLAGS.using_diff_in_mlstm
        model_specific_params['using_prod_in_mlstm'] = FLAGS.using_prod_in_mlstm
        model_specific_params['using_null_in_attention'] = FLAGS.using_null_in_attention
        model_specific_params['using_only_mlstm_feature'] = FLAGS.using_only_mlstm_feature
        # a model specific logger
        attlogger = logging.getLogger('spinn.attention')
        attlogger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(FLAGS.log_path, '{}.att.log'.format(FLAGS.experiment_name)))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        attlogger.addHandler(fh)
    else:
        raise Exception("Requested unimplemented model type %s" % FLAGS.model_type)

    # Build model.
    vocab_size = len(vocabulary)
    num_classes = len(data_manager.LABEL_MAP)
    ppdb_classes = len(load_ppdb.LABEL_MAP)
    if data_manager.SENTENCE_PAIR_DATA: # main dataset (not ppdb)
        trainer_cls = model_module.SentencePairTrainer
        model_cls = model_module.SentencePairModel
        use_sentence_pair = True
    else:
        ValueError('Only support sentence pair data, see also spin.data.')

    model = model_cls(model_dim=FLAGS.model_dim,
         word_embedding_dim=FLAGS.word_embedding_dim,
         vocab_size=vocab_size,
         initial_embeddings=initial_embeddings,
         mlp_dim=FLAGS.mlp_dim,
         embedding_keep_rate=FLAGS.embedding_keep_rate,
         classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
         tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
         transition_weight=FLAGS.transition_weight,
         encode_style=FLAGS.encode_style,
         encode_reverse=FLAGS.encode_reverse,
         encode_bidirectional=FLAGS.encode_bidirectional,
         encode_num_layers=FLAGS.encode_num_layers,
         use_sentence_pair=use_sentence_pair,
         use_skips=FLAGS.use_skips,
         lateral_tracking=FLAGS.lateral_tracking,
         use_tracking_in_composition=FLAGS.use_tracking_in_composition,
         use_difference_feature=FLAGS.use_difference_feature,
         use_product_feature=FLAGS.use_product_feature,
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
         model_specific_params=model_specific_params,
         main_mlp_num_classes=num_classes,
         ppdb_num_classes=ppdb_classes,
        )

    # Build optimizer.
    if FLAGS.optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    elif FLAGS.optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=FLAGS.learning_rate, eps=1e-08)
    else:
        raise NotImplementedError

    # Build trainer.
    classifier_trainer = trainer_cls(model, optimizer)

    standard_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.experiment_name)
    best_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.experiment_name, best=True)

    # Load checkpoint if available.
    if FLAGS.load_best and os.path.isfile(best_checkpoint_path):
        logger.Log("Found best checkpoint, restoring.")
        step, best_dev_error = classifier_trainer.load(best_checkpoint_path)
        logger.Log("Resuming at step: {} with best dev accuracy: {}".format(step, 1. - best_dev_error))
    elif os.path.isfile(standard_checkpoint_path):
        logger.Log("Found checkpoint, restoring.")
        step, best_dev_error = classifier_trainer.load(standard_checkpoint_path)
        logger.Log("Resuming at step: {} with best dev accuracy: {}".format(step, 1. - best_dev_error))
    else:
        assert not only_forward, "Can't run an eval-only run without a checkpoint. Supply a checkpoint."
        step = 1
        best_dev_error = 1.0

    # Print model size.
    logger.Log("Architecture: {}".format(model))
    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0) for w in model.parameters()])
    logger.Log("Total params: {}".format(total_params))

    # GPU support.
    the_gpu.gpu = FLAGS.gpu
    if FLAGS.gpu >= 0:
        model.cuda()
    else:
        model.cpu()

    # Debug
    def set_debug(self):
        self.debug = FLAGS.debug
    model.apply(set_debug)



    # Do an evaluation-only run.
    if only_forward:
        for index, eval_set in enumerate(eval_iterators):
            acc = evaluate(model, eval_set, logger, metrics_logger, step, vocabulary)
    else:
         # Train
        logger.Log("Training.")

        # New Training Loop
        progress_bar = SimpleProgressBar(msg="Training", bar_length=60, enabled=FLAGS.show_progress_bar)
        progress_bar.step(i=0, total=FLAGS.statistics_interval_steps)

        # Accumulate useful statistics.
        statis = Accumulator(maxlen=FLAGS.deque_length)

        for step in range(step, FLAGS.training_steps + 1):

            # set training mode, pytorch framework
            model.train()

            start = time.time()

            # get one batch from main training dataset
            X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = training_data_iter.next()

            # get one batch from ppdb
            ppdb_X_batch, ppdb_transitions_batch, ppdb_y_batch, ppdb_num_transitions_batch, ppdb_train_ids = ppdb_training_data_iter.next()

            total_tokens = sum([(n+1)/2 for n in num_transitions_batch.reshape(-1)]) \
                           + sum([(n+1)/2 for n in ppdb_num_transitions_batch.reshape(-1)])     # change it to
            statis.add('total_tokens', total_tokens)

            # Reset cached gradients.
            optimizer.zero_grad()

            # Run model with one batch of main dataset
            output = model(X_batch, transitions_batch,
                           True,    # main data set
                           y_batch=y_batch,
                           use_internal_parser=FLAGS.use_internal_parser,
                           validate_transitions=FLAGS.validate_transitions,)

            # Normalize output.
            logits = F.log_softmax(output)

            # run model with ppdb data set
            output_ppdb = model(ppdb_X_batch, ppdb_transitions_batch,
                                False,  # ppdb data set
                                y_batch=ppdb_y_batch,
                                use_internal_parser=FLAGS.use_internal_parser,
                                validate_transitions=FLAGS.validate_transitions,
                                )
            ppdb_logits = F.log_softmax(output_ppdb)

            # Calculate class accuracy.
            target = torch.from_numpy(y_batch).long()
            pred = logits.data.max(1)[1].cpu() # get the index of the max log-probability
            class_acc = pred.eq(target).sum() / float(target.size(0))

            # class accuracy of ppdb
            ppdb_target = torch.from_numpy(ppdb_y_batch).long()
            ppdb_pred = ppdb_logits.data.max(1)[1].cpu()
            ppdb_class_acc = ppdb_pred.eq(ppdb_target).sum() / float(ppdb_target.size(0))


            # Calculate class loss.
            xent_loss = nn.NLLLoss()(logits, to_gpu(Variable(target, volatile=False)))
            ppdb_xent_loss = nn.NLLLoss()(ppdb_logits, to_gpu(Variable(ppdb_target, volatile=False)))
            xent_cost_val = xent_loss.data[0]
            ppdb_xent_cost_val = ppdb_xent_loss.data[0]

            # Extract L2 Cost
            l2_loss = l2_cost(model, FLAGS.l2_lambda) if FLAGS.use_l2_cost else None
            l2_cost_val = l2_loss.data[0] if l2_loss else 0

            # Accumulate Total Loss Variable
            total_loss = 0.0
            total_loss += xent_loss
            total_loss += ppdb_xent_loss * FLAGS.ppdb_loss_weight   # should be tuned
            if l2_loss is not None: total_loss += l2_loss
            total_cost_val = total_loss.data[0]

            # keep metric
            statis.add('class_acc', class_acc)
            statis.add('ppdb_class_acc', ppdb_class_acc)

            # Backward pass.
            total_loss.backward()

            # Hard Gradient Clipping
            clip = FLAGS.clipping_max_value
            for p in model.parameters():
                if p.requires_grad:
                    p.grad.data.clamp_(min=-clip, max=clip)

            # Learning Rate Decay
            if FLAGS.actively_decay_learning_rate:
                optimizer.lr = FLAGS.learning_rate * (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))

            # Gradient descent step.
            optimizer.step()

            end = time.time()
            total_time = end - start

            statis.add('total_time', total_time)

            if step % FLAGS.statistics_interval_steps == 0:
                progress_bar.step(i=FLAGS.statistics_interval_steps, total=FLAGS.statistics_interval_steps)
                progress_bar.finish()
                avg_class_acc = statis.get_avg('class_acc')
                ppdb_avg_class_acc = statis.get_avg('ppdb_class_acc')
                time_metric = time_per_token(statis.get('total_tokens'), statis.get('total_time'))

                logstr = ''
                logstr += 'Step: {}'.format(step)
                logstr += ' Acc: {:.5f} {:.5f}'.format(avg_class_acc, ppdb_avg_class_acc)
                logstr += ' Cost(last step): {:.5f} {:.5f} {:.5f} {:.5f}'.format(total_cost_val, xent_cost_val, ppdb_xent_cost_val, l2_cost_val)
                logstr += ' Time: {:.5f}'.format(time_metric)
                logger.Log(logstr)

            if step % FLAGS.eval_interval_steps == 0:
                # do evaluation
                for index, eval_set in enumerate(eval_iterators):
                    acc = evaluate(model, eval_set, logger, metrics_logger, step)
                    if FLAGS.ckpt_on_best_dev_error and index == 0 and (1 - acc) < 0.99 * best_dev_error and step > FLAGS.ckpt_step:
                        best_dev_error = 1 - acc
                        logger.Log("Checkpointing with new best dev accuracy of %f" % acc)
                        classifier_trainer.save(best_checkpoint_path, step, best_dev_error)
                progress_bar.reset()

            if step % FLAGS.ckpt_step == 0:
                logger.Log("Checkpointing.")
                classifier_trainer.save(standard_checkpoint_path, step, best_dev_error)

            if step % FLAGS.statistics_interval_steps == 0:
                m_keys = statis.cache.keys()
                for k in m_keys:
                    metrics_logger.Log(k, statis.get_avg(k), step)
                statis = Accumulator(maxlen=FLAGS.deque_length)

            progress_bar.step(i=step % FLAGS.statistics_interval_steps, total=FLAGS.statistics_interval_steps)



if __name__ == '__main__':

    define_flags()

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

    # HACK: The "use_encode" flag will be deprecated. Instead use something like encode_style=LSTM.
    if FLAGS.use_encode:
        FLAGS.encode_style = "LSTM"

    run(only_forward=FLAGS.expanded_eval_only_mode)

