import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from spinn.util.misc import Args, Vocab, Example
from spinn.util.blocks import to_cpu, to_gpu
from spinn.util.blocks import Embed, MLP
from fat_stack import SPINN

class SentencePairTrainer():
    """
    required by the framework, fat_classifier.py @291,295
    init as classifier_trainer at fat_classifier.py @337
    """
    pass


class SPINNAttention(nn.module):

    def __init__(self):
        # default define, TODO
        pass

    def forward(self, premise_stack, hypothesis_stack):

        pass


class SentencePairModel(nn.module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 mlp_dim=None,
                 embedding_keep_rate=None,
                 classifier_keep_rate=None,     # TODO mean?
                 tracking_lstm_hidden_dim=4,    # TODO mean?
                 transition_weight=None,        # TODO mean?
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

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes,
            num_mlp_layers, mlp_bn, classifier_dropout_rate)

    def build_spinn(self, args, vocab, use_skips):
        return SPINN(args, vocab, use_skips=use_skips)

    def build_attention(self):
        pass

    def build_example(self, sentences, transitions):
        batch_size = sentences.shape[0]

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
        # TODO instead of return the final hidden vector, return the stack
        self.spinn.reset_state()
        state, transition_acc, transition_loss = self.spinn(example,
                               use_internal_parser=use_internal_parser,
                               validate_transitions=validate_transitions)
        return state, transition_acc, transition_loss

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
        h_m = self.attention(ps, hs)    # matching vector
        features = np.concat(hs[0], h_m)


        # output layer
        output = self.mlp(features)

        self.output_hook(output, sentences, transitions, y_batch)

        return output



class SentenceModel(nn.module):
    """
    required by the framework, fat_classifier.py@296
    init as model at fat_classifier.py@300
    because attention model take two sentences, this model might never be used
    """
    def __init__(self):
        raise Exception("")


