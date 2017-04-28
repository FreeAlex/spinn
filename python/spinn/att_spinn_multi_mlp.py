import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn.modules.rnn import RNNCellBase, LSTMCell
from torch.nn.parameter import Parameter
from spinn.util.misc import Args, Vocab, Example
from spinn.util.blocks import to_cpu, to_gpu, get_h
from spinn.util.blocks import Embed, MLP
from spinn.att_spinn import SPINNAttExt, AttentionModel
import logging

logger = logging.getLogger("spinn.attention")


class SentencePairTrainer():
    """
    required by the framework
    """
    def __init__(self, model, optimizer):
        logger.info('attspinn trainer init')
        self.model = model
        self.optimizer = optimizer

    def save(self, filename, step, best_dev_error):
        torch.save({
            'step': step,
            'best_dev_error': best_dev_error,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']

        # HACK: Compatability for saving supervised SPINN and loading RL SPINN.
        if 'baseline' in self.model.state_dict().keys() and 'baseline' not in model_state_dict:
            model_state_dict['baseline'] = torch.FloatTensor([0.0])

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step'], checkpoint['best_dev_error']

class SentencePairModel(nn.Module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 mlp_dim=None,
                 embedding_keep_rate=None,
                 classifier_keep_rate=None,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 use_encode=None,
                 encode_reverse=None,
                 encode_bidirectional=None,
                 encode_num_layers=None,
                 use_skips=False,
                 lateral_tracking=None,
                 use_tracking_in_composition=None,
                 # use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
                 num_mlp_layers=None,
                 mlp_bn=None,
                 model_specific_params={},
                 main_mlp_num_classes=None,
                 ppdb_num_classes=None,
                 **kwargs
                ):
        super(SentencePairModel, self).__init__()
        logger.info('ATTSPINN SentencePairModel init...')
        # self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature
        self.using_only_mlstm_feature = model_specific_params['using_only_mlstm_feature']

        self.hidden_dim = hidden_dim = model_dim / 2
        # features_dim = hidden_dim * 2 if use_sentence_pair else hidden_dim
        features_dim = hidden_dim
        if not self.using_only_mlstm_feature:
            logger.info('using not only matching lstm feature')
            features_dim += hidden_dim
            # [premise, hypothesis, diff, product]
            if self.use_difference_feature:
                logger.info('using diff feature in MLP')
                features_dim += self.hidden_dim
            if self.use_product_feature:
                logging.info('using prod feature in MLP')
                features_dim += self.hidden_dim

        mlp_input_dim = features_dim

        self.initial_embeddings = initial_embeddings
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        classifier_dropout_rate = 1. - classifier_keep_rate

        args = Args()
        args.lateral_tracking = lateral_tracking
        args.use_tracking_in_composition = use_tracking_in_composition
        args.size = model_dim/2
        args.tracker_size = tracking_lstm_hidden_dim
        args.transition_weight = transition_weight
        args.using_diff_in_mlstm = model_specific_params['using_diff_in_mlstm']
        args.using_prod_in_mlstm = model_specific_params['using_prod_in_mlstm']
        args.using_null_in_attention = model_specific_params['using_null_in_attention']

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        # The input embeddings represent the hidden and cell state, so multiply by 2.
        self.embedding_dropout_rate = 1. - embedding_keep_rate
        input_embedding_dim = args.size * 2

        # Create dynamic embedding layer.
        self.embed = Embed(input_embedding_dim, vocab.size, vectors=vocab.vectors)

        self.use_encode = use_encode
        if use_encode:
            self.encode_reverse = encode_reverse
            self.encode_bidirectional = encode_bidirectional
            self.bi = 2 if self.encode_bidirectional else 1
            self.encode_num_layers = encode_num_layers
            self.encode = nn.LSTM(model_dim, model_dim / self.bi, num_layers=encode_num_layers,
                batch_first=True,
                bidirectional=self.encode_bidirectional,
                dropout=self.embedding_dropout_rate)

        self.spinn = self.build_spinn(args, vocab, use_skips)

        self.attention = self.build_attention(args)

        # init MLP layer
        self.main_mlp = MLP(mlp_input_dim, mlp_dim, main_mlp_num_classes, num_mlp_layers, mlp_bn, classifier_dropout_rate)
        self.ppdb_mlp = MLP(mlp_input_dim, mlp_dim, ppdb_num_classes, num_mlp_layers, mlp_bn, classifier_dropout_rate)


    def build_spinn(self, args, vocab, use_skips):
        return SPINNAttExt(args, vocab, use_skips=use_skips)

    def build_attention(self, args):
        return AttentionModel(args)

    def build_example(self, sentences, transitions):
        batch_size = sentences.shape[0]
        # sentences: (#batches, #feature, #2)
        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        # Build Transitions
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def run_spinn(self, example, use_internal_parser, validate_transitions=True):
        # TODO xz. instead of return the final hidden vector, return the stack
        self.spinn.reset_state()
        state, transition_acc, transition_loss = self.spinn(example,
                               use_internal_parser=use_internal_parser,
                               validate_transitions=validate_transitions)
        premise_stack, hypothesis_stack = self.spinn.get_h_stacks()

        #state: a batch of stack [stack_1, ..., stack_n] where n is batch size
        return premise_stack, hypothesis_stack, transition_acc, transition_loss

    def output_hook(self, output, sentences, transitions, y_batch=None, main_or_ppdb=None):
        pass


    def forward(self, sentences, transitions, main_or_ppdb, y_batch=None,
                use_internal_parser=False, validate_transitions=True):

        example = self.build_example(sentences, transitions)    # example = [premises, hypothesis]

        b, l = example.tokens.size()[:2]

        embeds = self.embed(example.tokens)
        embeds = F.dropout(embeds, self.embedding_dropout_rate, training=self.training)
        embeds = torch.chunk(to_cpu(embeds), b, 0)

        if self.use_encode:
            to_encode = torch.cat([e.unsqueeze(0) for e in embeds], 0)
            encoded = self.run_encode(to_encode)
            embeds = [x.squeeze() for x in torch.chunk(encoded, b, 0)]

        # Make Buffers
        embeds = [torch.chunk(x, l, 0) for x in embeds]
        buffers = [list(reversed(x)) for x in embeds]

        example.bufs = buffers

        # Premise stack & hypothesis stack
        ps, hs, transition_acc, transition_loss = self.run_spinn(example, use_internal_parser, validate_transitions)

        if use_internal_parser:
            self.transition_acc = transition_acc
            self.transition_loss = transition_loss
        else:
            assert transition_acc is None or transition_acc == 0, transition_acc
            assert transition_loss is None or transition_loss == 0, transition_loss

        # attention model
        h_m = self.attention(ps, hs)    # matching matrix batch_size * hidden_dim
        assert h_m.size() == (len(ps), self.hidden_dim)
        # print 'run attention complete'

        features = self.build_features(hs, h_m)

        # output layer
        if main_or_ppdb:
            output = self.main_mlp(features)
        else:
            output = self.ppdb_mlp(features)

        self.output_hook(output, sentences, transitions, y_batch, main_or_ppdb)
        # print 'one batch complete, output size', output.size()
        return output

    def build_features(self, hstacks, h_m):
        h_ks = [stack[-1].unsqueeze(0) for stack in hstacks]
        h_ks = torch.cat(h_ks, 0) # extract the final representation from each stack
        assert h_ks.size(0) == h_m.size(0)
        assert h_ks.size(1) == self.hidden_dim
        assert h_m.size(1) == self.hidden_dim

        features = [h_m]
        if not self.using_only_mlstm_feature:
            features.append(h_ks)
            if self.use_difference_feature:
                features.append(h_ks - h_m)
            if self.use_product_feature:
                features.append(h_ks * h_m)
        features = torch.cat(features, 1) # D0 -> batch, D1 -> representation vector
        return features

    def set_recording_attention_weight_matrix(self, flag=True):
        self.attention.set_recording_attention_weight_matrix(flag)

    def get_attention_matrix_from_last_forward(self):
        return self.attention.attention_weight_matrix








