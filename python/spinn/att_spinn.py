import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter
from spinn.util.misc import Args, Vocab, Example
from spinn.util.blocks import to_cpu, to_gpu, get_h
from spinn.util.blocks import Embed, MLP
from fat_stack import SPINN
from itertools import izip
import math

class SentencePairTrainer():
    """
    required by the framework, fat_classifier.py @291,295
    init as classifier_trainer at fat_classifier.py @337
    """
    def __init__(self, model, optimizer):
        print 'attspinn trainer init'
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

class SPINNAttExt(SPINN):

    def __init__(self, args, vocab, use_skips):
        print 'SPINNAttExt init...'
        super(SPINNAttExt, self).__init__(args, vocab, use_skips)
        self.hidden_dim = args.size
        self.states = [] # only support one example now, premise and hypothesis
        self.debug = False

    def forward_hook(self):
        batch_size = len(self.bufs)
        self.states = [[] for i in range(batch_size)]
        # print 'forward hook, clear states'

    def shift_phase_hook(self, tops, trackings, stacks, idxs):
        # print 'shift_phase_hook...'
        for idx, stack in izip(idxs, stacks):
            h = get_h(stack[-1], self.hidden_dim)
            assert h.size() == torch.Size([1, self.hidden_dim]), 'hsize: {}'.format(h.size())
            self.states[idx].append(h)

    def reduce_phase_hook(self, lefts, rights, trackings, reduce_stacks, r_idxs=None):

        for idx, stack in izip(r_idxs, reduce_stacks):
            h = get_h(stack[-1], self.hidden_dim)
            assert h.size() == torch.Size([1, self.hidden_dim]), 'hsize: {}'.format(h.size())
            self.states[idx].append(h)

    def get_hidden_stacks(self):
        batch_size = len(self.states) / 2
        premise_stacks = self.states[:batch_size]
        hyphothesis_stacks = self.states[batch_size:]

        return premise_stacks, hyphothesis_stacks

    def get_h_stacks(self):
        premise_stacks, hypothesis_stacks = self.get_hidden_stacks()
        assert len(premise_stacks) == len(hypothesis_stacks)
        pstacks = [torch.cat(ps, 0) for ps in premise_stacks]
        hstacks = [torch.cat(hs, 0) for hs in hypothesis_stacks]
        if self.debug:
            print 'pstack size:', [ps.size() for ps in pstacks]
            print 'hstack size:', [hs.size() for hs in hstacks]
        return pstacks, hstacks



class MatchingLSTMCell(RNNCellBase):
    # TODO: xz. modify this description, this is copied from LSTMCell
    r"""A long short-term memory (LSTM) cell.
    .. math::
        \begin{array}{ll}
        i = sigmoid(W_{mi} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = sigmoid(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = sigmoid(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c_t) \\
        \end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: True
    """

    def __init__(self, input_size, hidden_size):
        super(MatchingLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 4 for c, i, f, o
        self.weight_ih = Parameter(torch.Tensor(4*hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4*hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(4*hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx, cx):
        gates = self.weight_ih.mv(input) + self.weight_hh.mv(hx) + self.bias_ih
        cgate, igate, fgate, ogate = gates.chunk(4, 0)
        # non linear
        cgate = F.tanh(cgate)
        igate = F.sigmoid(igate)
        fgate = F.sigmoid(fgate)
        ogate = F.sigmoid(ogate)

        cy = (fgate * cx) + (igate * cgate)
        hy = ogate * F.tanh(cy)
        return hy, cy

class AttentionModel(nn.Module):

    def __init__(self, args):
        super(AttentionModel, self).__init__()
        self.hidden_dim = args.size #
        self.matching_input_size = args.size * 2
        self.matching_lstm_unit = MatchingLSTMCell(self.matching_input_size, self.hidden_dim)
        self.w_e = Parameter(torch.Tensor(self.hidden_dim))
        self.weight_premise = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.weight_hypothesis = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.weight_matching = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        print 'AttentionModel init'
        self.reset_parameters()

    def matching_lstm(self, mk, hmk_1, cmk_1):
        hmk, cmk = self.matching_lstm_unit(mk, hmk_1, cmk_1)
        return hmk, cmk

    def attention_vector(self, ps, hk, hmk_1):
        # ak = softmax([... ekj ...]
        # ekj = w^e * tanh(W_pP + W_h*h + W_m*h_m)
        fe_p = F.linear(ps, self.weight_premise)
        fe_h = self.weight_hypothesis.mv(hk)
        fe_m = self.weight_matching.mv(hmk_1)
        for i in range(fe_p.size()[0]):
            fe_p[i] = fe_p[i] + fe_h + fe_m
        ak = F.softmax(F.tanh(fe_p).mv(self.w_e))
        return ak

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def run_one_pair(self, ps, hs):

        hmk_1 = Variable(to_gpu(torch.zeros(self.hidden_dim)), volatile=not self.training)
        cmk_1 = Variable(to_gpu(torch.zeros(self.hidden_dim)), volatile=not self.training)
        # print 'hs size', hs.size()
        for hk in hs:
            # for each step
            ak = self.attention_vector(ps, hk, hmk_1)
            # print 'ak size', ak.size()
            pmk = ps.t().mv(ak)
            mk = torch.cat([pmk, hk], 0)
            hmk_1, cmk_1 = self.matching_lstm(mk, hmk_1, cmk_1)
        return hmk_1

    def forward(self, premise_stacks, hypothesis_stacks):

        assert len(premise_stacks) == len(hypothesis_stacks)

        hms = []
        # print 'size of stacks', len(premise_stacks)
        for ps, hs in izip(premise_stacks, hypothesis_stacks):
            hm = self.run_one_pair(ps, hs)
            # print 'run one pair complete'
            hms.append(hm.unsqueeze(0))

        hms = torch.cat(hms, 0) # batch it
        return hms


class SentencePairModel(nn.Module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
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
                 **kwargs
                ):
        super(SentencePairModel, self).__init__()
        print 'ATTSPINN SentencePairModel init...'
        # self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature

        self.hidden_dim = hidden_dim = model_dim / 2
        # features_dim = hidden_dim * 2 if use_sentence_pair else hidden_dim
        features_dim = model_dim

        # [premise, hypothesis, diff, product]
        if self.use_difference_feature:
            features_dim += self.hidden_dim
        if self.use_product_feature:
            features_dim += self.hidden_dim

        mlp_input_dim = features_dim

        self.initial_embeddings = initial_embeddings
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        classifier_dropout_rate = 1. - classifier_keep_rate # TODO mean?

        args = Args()
        args.lateral_tracking = lateral_tracking
        args.use_tracking_in_composition = use_tracking_in_composition
        args.size = model_dim/2
        args.tracker_size = tracking_lstm_hidden_dim
        args.transition_weight = transition_weight

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

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes,
            num_mlp_layers, mlp_bn, classifier_dropout_rate)

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

    def output_hook(self, output, sentences, transitions, y_batch=None):
        pass


    def forward(self, sentences, transitions, y_batch=None,
                use_internal_parser=False, validate_transitions=True):
        example = self.build_example(sentences, transitions)

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

        self.transition_acc = transition_acc
        self.transition_loss = transition_loss

        # attention model
        h_m = self.attention(ps, hs)    # matching matrix batch_size * hidden_dim
        assert h_m.size(1) == self.hidden_dim
        # print 'run attention complete'

        features = self.build_features(hs, h_m)

        # output layer
        output = self.mlp(features)

        self.output_hook(output, sentences, transitions, y_batch)
        # print 'one batch complete, output size', output.size()
        return output

    def build_features(self, hstacks, h_m):
        h_ks = [stack[-1].unsqueeze(0) for stack in hstacks]
        h_ks = torch.cat(h_ks, 0) # extract the final representation from each stack
        assert h_ks.size(0) == h_m.size(0)
        assert h_ks.size(1) == self.hidden_dim
        assert h_m.size(1) == self.hidden_dim

        features = [h_ks, h_m]
        if self.use_difference_feature:
            features.append(h_ks - h_m)
        if self.use_product_feature:
            features.append(h_ks * h_m)
        features = torch.cat(features, 1) # D0 -> batch, D1 -> representation vector
        return features



class SentenceModel(nn.Module):
    """
    required by the framework, fat_classifier.py@296
    init as model at fat_classifier.py@300
    because attention model take two sentences, this model might never be used
    """
    def __init__(self):
        raise Exception("")


