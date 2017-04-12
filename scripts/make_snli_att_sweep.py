# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np
import gflags
import sys

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"
SS_SET = "SS_SET"

FLAGS = gflags.FLAGS

gflags.DEFINE_string("training_data_path", "/home/xz1364/mlworkspace/snli_1.0/snli_1.0_train.jsonl", "")
gflags.DEFINE_string("eval_data_path", "/home/xz1364/mlworkspace/snli_1.0/snli_1.0_dev.jsonl", "")
gflags.DEFINE_string("embedding_data_path", "/home/xz1364/mlworkspace/glove/glove.840B.300d.txt", "")
gflags.DEFINE_string("log_path", ".", "")
gflags.DEFINE_string("exp_names", 'dual-0', "experiment names, seperate by coma(,)")
gflags.DEFINE_string("slurm_name", 'dual-0', 'the file name of slurm output and err file')
gflags.DEFINE_integer("mem", 10, "memory should be used for hpc for each task")
gflags.DEFINE_string("spinn_path", '/home/xz1364/repos/faspinn-dev3/python', 'the model path so that spinn can run')
gflags.DEFINE_bool("using_diff_in_mlstm", True, 'wether or not use diff feature in mlstm')
gflags.DEFINE_bool('using_prod_in_mlstm', True, 'wether or not use prod feature in mlstm')
gflags.DEFINE_bool('using_null_in_attention', True, 'wether using null vector in attention, so that weights can assign to null vector')
gflags.DEFINE_bool('using_only_mlstm_feature', False, 'wether using only mlstm feature in final represenation')
gflags.DEFINE_bool('using_dual_attention', True, 'wether using dual attention')
gflags.DEFINE_bool('gpu', True, 'wether use GPU or not')
FLAGS(sys.argv)

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "data_type":     "snli",
    "model_type":      "ATTSPINN",
    "training_data_path":    FLAGS.training_data_path,
    "eval_data_path":    FLAGS.eval_data_path,
    "embedding_data_path": FLAGS.embedding_data_path,
    "log_path": FLAGS.log_path,
    "metrics_path": FLAGS.log_path,
    "ckpt_path":  FLAGS.log_path,
    "word_embedding_dim":   "300",
    "model_dim":   "600",
    "seq_length":   "150",
    "eval_seq_length":  "150",
    "eval_interval_steps": "1000",
    "statistics_interval_steps": "1000",
    "batch_size":  "64",
    "num_mlp_layers": "2",
    "use_difference_feature": "",
    "use_product_feature": "",
}
# deal with confiurable fixed parameters
if FLAGS.using_diff_in_mlstm:
    FIXED_PARAMETERS['using_diff_in_mlstm'] = ''
else:
    FIXED_PARAMETERS['nousing_diff_in_mlstm'] = ''

if FLAGS.using_prod_in_mlstm:
    FIXED_PARAMETERS['using_prod_in_mlstm'] = ''
else:
    FIXED_PARAMETERS['nousing_prod_in_mlstm'] = ''

if FLAGS.using_null_in_attention:
    FIXED_PARAMETERS['using_null_in_attention'] = ''
else:
    FIXED_PARAMETERS['nousing_null_in_attention'] = ''

if FLAGS.using_only_mlstm_feature:
    FIXED_PARAMETERS['using_only_mlstm_feature'] = ''
else:
    FIXED_PARAMETERS['nousing_only_mlstm_feature'] = ''

if FLAGS.using_dual_attention:
    FIXED_PARAMETERS['using_dual_attention'] = ''
else:
    FIXED_PARAMETERS['nousing_dual_attention'] = ''

# Tunable parameters.
SWEEP_PARAMETERS = {
    "learning_rate":      ("lr", EXP, 0.0001, 0.0008),  # RNN likes higher, but below 009.
    "l2_lambda":          ("l2", EXP, 8e-7, 2e-5),
    "semantic_classifier_keep_rate": ("skr", LIN, 0.7, 0.95),  # NB: Keep rates may depend considerably on dims.
    "embedding_keep_rate": ("ekr", LIN, 0.7, 0.95),
    "learning_rate_decay_per_10k_steps": ("dec", EXP, 0.5, 1.0),
    # 'mlp_dim': ('mlpd', SS_SET, [512], 0),   # 0 is useless
    # 'model_dim': ('mdim', SS_SET, [300, 600], 0),  # 0 is useless
    # 'num_mlp_layers': ('mlpl', SS_SET, [1, 2], 0),
}

exp_names = FLAGS.exp_names.split(',')


def print_script_head():
    print '''#!/bin/sh
# Generic job script for all experiments.
#SBATCH --time=24:00:00
#SBATCH --reservation=mhealth'''
    print '#SBATCH --mem={}GB'.format(FLAGS.mem * len(exp_names))
    print '#SBATCH --output=slurm-{}-%j.out'.format(FLAGS.slurm_name)
    print '#SBATCH --error=slurm-{}-%j.err'.format(FLAGS.slurm_name)
    print '#SBATCH --job-name={}'.format(FLAGS.slurm_name)
    if FLAGS.gpu:
        print '#SBATCH --partition=gpu'
        print '#SBATCH --gres=gpu:1'
        print '#PRINCE PRINCE_GPU_COMPUTE_MODE=default'
    print '\n'
    print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
    print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
    print '''
# Make sure we have access to HPC-managed libraries.
module load python/intel/2.7.12 pytorch/intel/20170226 protobuf/intel/3.1.0
# Make sure the required pacage has been installed
pip install --user python-gflags==2.0
    '''
    print 'export PYTHONPATH=$PYTHONPATH:{}'.format(FLAGS.spinn_path)
    print '\n'

def get_fixed_params():
    params = FIXED_PARAMETERS
    return params

def get_sweep_params():
    params = {}

    for param in SWEEP_PARAMETERS:
        config = SWEEP_PARAMETERS[param]
        t = config[1]
        mn = config[2]
        mx = config[3]

        r = random.uniform(0, 1)
        if t == EXP:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = np.exp(lmn + (lmx - lmn) * r)
        elif t == SS_BASE:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = 1 - np.exp(lmn + (lmx - lmn) * r)
        elif t == SS_SET:
            sample = random.choice(config[2])
        else:
            sample = mn + (mx - mn) * r

        if isinstance(mn, int):
            sample = int(round(sample, 0))
            val_disp = str(sample)
        else:
            val_disp = "%.2g" % sample

        params[param] = sample

    return params



print_script_head()

for exp_name in exp_names:
    params = {}
    print 'echo running experiment {}'.format(exp_name)
    print 'python -m spinn.models.fat_classifier  --noshow_progress_bar \\'
    if FLAGS.gpu:
        print ' --gpu 0 \\'
    print ' --experiment_name {} \\'.format(exp_name)
    # update method will replace fixed params if it's in sweep params, BE CAREFUL if change the order
    params.update(get_fixed_params())
    params.update(get_sweep_params())
    for param in params:
        value = params[param]
        print " --" + param + " " + str(value) + " \\"
    print ' & \\'
    print '\n'

print ''
print 'echo waiting'
print 'wait'