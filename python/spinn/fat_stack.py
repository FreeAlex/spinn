import itertools

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import BaseSentencePairTrainer, Reduce
from spinn.util.blocks import LSTMState, Embed, MLP, Linear, LSTM
from spinn.util.blocks import reverse_tensor
from spinn.util.blocks import bundle, unbundle, to_cpu, to_gpu, treelstm, lstm
from spinn.util.blocks import get_h, get_c
from spinn.util.misc import Args, Vocab, Example
from spinn.util.blocks import HeKaimingInitializer


T_SKIP   = 2
T_SHIFT  = 0
T_REDUCE = 1


"""

Missing Features

- [ ] Optionally use cell when predicting transitions.


"""


class SentencePairTrainer(BaseSentencePairTrainer): pass


class SentenceTrainer(SentencePairTrainer): pass


class Tracker(nn.Module):

    def __init__(self, size, tracker_size, lateral_tracking=True):
        super(Tracker, self).__init__()

        # Initialize layers.
        self.buf = Linear()(size, 4 * tracker_size, bias=False)
        self.stack1 = Linear()(size, 4 * tracker_size, bias=False)
        self.stack2 = Linear()(size, 4 * tracker_size, bias=False)

        if lateral_tracking:
            self.lateral = Linear(initializer=HeKaimingInitializer)(tracker_size, 4 * tracker_size)
        else:
            self.transform = Linear(initializer=HeKaimingInitializer)(4 * tracker_size, tracker_size)

        self.lateral_tracking = lateral_tracking
        self.state_size = tracker_size

        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def forward(self, top_buf, top_stack_1, top_stack_2):
        tracker_inp = self.buf(top_buf.h)
        tracker_inp += self.stack1(top_stack_1.h)
        tracker_inp += self.stack2(top_stack_2.h)

        batch_size = tracker_inp.size(0)

        if self.lateral_tracking:
            if self.h is not None:
                tracker_inp += self.lateral(self.h)
            if self.c is None:
                self.c = to_gpu(Variable(torch.from_numpy(
                    np.zeros((batch_size, self.state_size),
                                  dtype=np.float32)),
                    volatile=tracker_inp.volatile))

            # Run tracking lstm.
            self.c, self.h = lstm(self.c, tracker_inp)

            return self.h, self.c
        else:
            outp = self.transform(tracker_inp)
            return outp, None

    @property
    def states(self):
        return unbundle((self.c, self.h))

    @states.setter
    def states(self, state_iter):
        if state_iter is not None:
            state = bundle(state_iter)
            self.c, self.h = state.c, state.h


class SPINN(nn.Module):

    def __init__(self, args, vocab, use_skips=False):
        super(SPINN, self).__init__()

        # Optional debug mode.
        self.debug = False

        self.transition_weight = args.transition_weight
        self.use_skips = use_skips

        # Reduce function for semantic composition.
        self.reduce = Reduce(args.size, args.tracker_size, args.use_tracking_in_composition)
        if args.tracker_size is not None:
            self.tracker = Tracker(args.size, args.tracker_size, args.lateral_tracking)
            if args.transition_weight is not None:
                # TODO: Might be interesting to try a different network here.
                self.transition_net = nn.Linear(args.tracker_size, 3 if use_skips else 2)

        # Predict 2 or 3 actions depending on whether SKIPs will be predicted.
        choices = [T_SHIFT, T_REDUCE, T_SKIP] if use_skips else [T_SHIFT, T_REDUCE]
        self.choices = np.array(choices, dtype=np.int32)

    def reset_state(self):
        self.memories = []

    def forward(self, example, use_internal_parser=False, validate_transitions=True):
        self.buffers_n = (example.tokens.data != 0).long().sum(1).view(-1).tolist()

        if self.debug:
            seq_length = example.tokens.size(1)
            assert all(buf_n <= (seq_length + 1) // 2 for buf_n in self.buffers_n), \
                "All sentences (including cropped) must be the appropriate length."

        self.bufs = example.bufs

        # Notes on adding zeros to bufs/stacks.
        # - After the buffer is consumed, we need one zero on the buffer
        #   used as input to the tracker.
        # - For the first two steps, the stack would be empty, but we add
        #   zeros so that the tracker still gets input.
        zeros = to_gpu(Variable(torch.from_numpy(
            np.zeros(self.bufs[0][0].size(), dtype=np.float32)),
            volatile=self.bufs[0][0].volatile))

        # Trim unused tokens.
        self.bufs = [[zeros] + b[-b_n:] for b, b_n in zip(self.bufs, self.buffers_n)]

        self.stacks = [[zeros, zeros] for buf in self.bufs]

        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        if not hasattr(example, 'transitions'):
            # TODO: Support no transitions. In the meantime, must at least pass dummy transitions.
            raise ValueError('Transitions must be included.')
        self.forward_hook()
        return self.run(example.transitions,
                        run_internal_parser=True,
                        use_internal_parser=use_internal_parser,
                        validate_transitions=validate_transitions)

    def forward_hook(self):
        pass

    def validate(self, transitions, preds, stacks, bufs, zero_padded=True):
        # Note: There is one zero added to bufs, and two zeros added to stacks.
        # Make sure to adjust for this if using lengths of either.
        buf_adjust = 1 if zero_padded else 0
        stack_adjust = 2 if zero_padded else 0

        _transitions = np.array(transitions)

        # Fixup predicted skips.
        if len(self.choices) > 2:
            raise NotImplementedError("Can only validate actions for 2 choices right now.")

        buf_lens = [len(buf) - buf_adjust for buf in bufs]
        stack_lens = [len(stack) - stack_adjust for stack in stacks]

        # Cannot reduce on too small a stack
        must_shift = np.array([length < 2 for length in stack_lens])
        preds[must_shift] = T_SHIFT

        # Cannot shift on too small buf
        must_reduce = np.array([length < 1 for length in buf_lens])
        preds[must_reduce] = T_REDUCE

        # If the given action is skip, then must skip.
        preds[_transitions == T_SKIP] = T_SKIP

        return preds

    def predict_actions(self, transition_output, cant_skip):
        transition_dist = F.log_softmax(transition_output)
        transition_dist = transition_dist.data.cpu().numpy()
        transition_preds = transition_dist.argmax(axis=1)
        return transition_preds

    def get_statistics(self):
        # TODO: These are not necessarily the most efficient flatten operations...

        t_preds = np.array(reduce(lambda x, y: x + y.tolist(),
            [m["t_preds"] for m in self.memories], []))
        t_given = np.array(reduce(lambda x, y: x + y.tolist(),
            [m["t_given"] for m in self.memories], []))
        t_mask = np.array(reduce(lambda x, y: x + y.tolist(),
            [m["t_mask"] for m in self.memories], []))
        t_logits = [m["t_logits"] for m in self.memories]
        if len(t_logits) > 0:
            t_logits = torch.cat(t_logits, 0)

        return t_preds, t_logits, t_given, t_mask

    def get_transition_preds_per_example(self):
        t_preds, t_logits, t_given, t_mask = self.get_statistics()

        batch_size = t_mask.max()
        preds = []
        for batch_idx in range(batch_size):
            preds.append(t_preds[t_mask == batch_idx])

        return np.array(preds)

    def t_shift(self, buf, stack, tracking, buf_tops, trackings):
        """SHIFT: Should dequeue buffer and item to stack."""
        buf_tops.append(buf.pop())
        trackings.append(tracking)

    def t_reduce(self, buf, stack, tracking, lefts, rights, trackings):
        """REDUCE: Should compose top two items of the stack into new item."""

        # The right-most input will be popped first.
        for reduce_inp in [rights, lefts]:
            if len(stack) > 0:
                reduce_inp.append(stack.pop())
            else:
                if self.debug:
                    raise IndexError
                # If we try to Reduce, but there are less than 2 items on the stack,
                # then treat any available item as the right input, and use zeros
                # for any other inputs.
                # NOTE: Only happens on cropped data.
                zeros = to_gpu(Variable(
                    torch.from_numpy(np.zeros(buf[0].size(), dtype=np.float32)),
                    volatile=buf[0].volatile))
                reduce_inp.append(zeros)

        trackings.append(tracking)

    def t_skip(self):
        """SKIP: Acts as padding and is a noop."""
        pass

    def shift_phase(self, tops, trackings, stacks, idxs):
        """SHIFT: Should dequeue buffer and item to stack."""
        if len(stacks) > 0:
            shift_candidates = iter(tops)
            for stack in stacks:
                new_stack_item = next(shift_candidates)
                stack.append(new_stack_item)
    def shift_phase_hook(self, tops, trackings, stacks, idxs):
        pass

    def reduce_phase(self, lefts, rights, trackings, stacks):
        if len(stacks) > 0:
            reduced = iter(self.reduce(
                lefts, rights, trackings))
            for stack in stacks:
                new_stack_item = next(reduced)
                stack.append(new_stack_item)

    def reduce_phase_hook(self, lefts, rights, trackings, reduce_stacks):
        pass

    def loss_phase_hook(self):
        pass

    def run(self, inp_transitions, run_internal_parser=False, use_internal_parser=False, validate_transitions=True):
        transition_loss = None
        transition_acc = 0.0
        num_transitions = inp_transitions.shape[1]

        # Transition Loop
        # ===============

        for t_step in range(num_transitions):
            transitions = inp_transitions[:, t_step]
            transition_arr = list(transitions)
            sub_batch_size = len(transition_arr)

            # A mask to select all non-SKIP transitions.
            cant_skip = np.array([t != T_SKIP for t in transitions])

            # Remember important details from this time step.
            self.memory = {}

            # Run if:
            # A. We have a tracking component and,
            # B. There is at least one transition that will not be skipped.
            if hasattr(self, 'tracker') and (self.use_skips or sum(cant_skip) > 0):

                # Prepare tracker input.
                try:
                    top_buf = bundle(buf[-1] for buf in self.bufs)
                    top_stack_1 = bundle(stack[-1] for stack in self.stacks)
                    top_stack_2 = bundle(stack[-2] for stack in self.stacks)
                except:
                    # To elaborate on this exception, when cropping examples it is possible
                    # that your first 1 or 2 actions is a reduce action. It is unclear if this
                    # is a bug in cropping or a bug in how we think about cropping. In the meantime,
                    # turn on the truncate batch flag, and set the eval_seq_length very high.
                    raise NotImplementedError("Warning: You are probably trying to encode examples"
                          "with cropped transitions. Although, this is a reasonable"
                          "feature, when predicting/validating transitions, you"
                          "probably will not get the behavior that you expect. Disable"
                          "this exception if you dare.")
                    # Uncomment to handle weirdly placed actions like discussed in the above exception.
                    # =========
                    # zeros = to_gpu(Variable(torch.from_numpy(
                    #     np.zeros(self.bufs[0][0].size(), dtype=np.float32)),
                    #     volatile=self.bufs[0][0].volatile))
                    # top_buf = bundle(buf[-1] for buf in self.bufs)
                    # top_stack_1 = bundle(stack[-1] if len(stack) > 0 else zeros for stack in self.stacks)
                    # top_stack_2 = bundle(stack[-2] if len(stack) > 1 else zeros for stack in self.stacks)

                # Get hidden output from the tracker. Used to predict transitions.
                tracker_h, tracker_c = self.tracker(top_buf, top_stack_1, top_stack_2)

                if hasattr(self, 'transition_net'):
                    transition_output = self.transition_net(tracker_h)

                if hasattr(self, 'transition_net') and run_internal_parser:

                    # Predict Actions
                    # ===============

                    t_logits = F.log_softmax(transition_output)
                    t_given = transitions
                    # TODO: Mask before predicting. This should simplify things and reduce computation.
                    # The downside is that in the Action Phase, need to be smarter about which stacks/bufs
                    # are selected.
                    transition_preds = self.predict_actions(transition_output, cant_skip)

                    # Constrain to valid actions
                    # ==========================

                    if validate_transitions:
                        transition_preds = self.validate(transition_arr, transition_preds, self.stacks, self.bufs)

                    t_preds = transition_preds

                    # Indices of examples that have a transition.
                    t_mask = np.arange(sub_batch_size)

                    # Filter to non-SKIP values
                    # =========================

                    if not self.use_skips:
                        t_preds = t_preds[cant_skip]
                        t_given = t_given[cant_skip]
                        t_mask = t_mask[cant_skip]

                        # Be careful when filtering distributions. These values are used to
                        # calculate loss and need to be used in backprop.
                        index = (cant_skip * np.arange(cant_skip.shape[0]))[cant_skip]
                        index = to_gpu(Variable(torch.from_numpy(index).long(), volatile=t_logits.volatile))
                        t_logits = torch.index_select(t_logits, 0, index)


                    # Memories
                    # ========
                    # Keep track of key values to determine accuracy and loss.
                    # (optional) Filter to only non-skipped transitions. When filtering values
                    # that will be backpropagated over, be careful that gradient flow isn't broken.

                    # Actual transition predictions. Used to measure transition accuracy.
                    self.memory["t_preds"] = t_preds

                    # Distribution of transitions use to calculate transition loss.
                    self.memory["t_logits"] = t_logits

                    # Given transitions.
                    self.memory["t_given"] = t_given

                    # Record step index.
                    self.memory["t_mask"] = t_mask

                    # TODO: Write tests to make sure memories look right in the various settings.

                    # If this FLAG is set, then use the predicted actions rather than the given.
                    if use_internal_parser:
                        transition_arr = transition_preds.tolist()

            # Pre-Action Phase
            # ================

            # For SHIFT
            s_stacks, s_tops, s_trackings, s_idxs = [], [], [], []

            # For REDUCE
            r_stacks, r_lefts, r_rights, r_trackings, r_idxs = [], [], [], [], []

            batch = zip(transition_arr, self.bufs, self.stacks,
                        self.tracker.states if hasattr(self, 'tracker') and self.tracker.h is not None
                        else itertools.repeat(None))

            for batch_idx, (transition, buf, stack, tracking) in enumerate(batch):
                if transition == T_SHIFT: # shift
                    self.t_shift(buf, stack, tracking, s_tops, s_trackings)
                    s_idxs.append(batch_idx)
                    s_stacks.append(stack)
                elif transition == T_REDUCE: # reduce
                    self.t_reduce(buf, stack, tracking, r_lefts, r_rights, r_trackings)
                    r_stacks.append(stack)
                    r_idxs.append(batch_idx)
                elif transition == T_SKIP: # skip
                    self.t_skip()

            # Action Phase
            # ============

            self.shift_phase(s_tops, s_trackings, s_stacks, s_idxs)
            self.shift_phase_hook(s_tops, s_trackings, s_stacks, s_idxs)
            self.reduce_phase(r_lefts, r_rights, r_trackings, r_stacks)
            self.reduce_phase_hook(r_lefts, r_rights, r_trackings, r_stacks, r_idxs=r_idxs)

            # Memory Phase
            # ============

            self.memories.append(self.memory)

        # Loss Phase
        # ==========

        if hasattr(self, 'tracker') and hasattr(self, 'transition_net'):
            t_preds, t_logits, t_given, _ = self.get_statistics()

            # We compute accuracy and loss after all transitions have complete,
            # since examples can have different lengths when not using skips.
            transition_acc = (t_preds == t_given).sum() / float(t_preds.shape[0])
            transition_loss = nn.NLLLoss()(t_logits, to_gpu(Variable(
                torch.from_numpy(t_given), volatile=t_logits.volatile)))
            transition_loss *= self.transition_weight

        self.loss_phase_hook()

        if self.debug:
            assert all(len(stack) == 3 for stack in self.stacks), \
                "Stacks should be fully reduced and have 3 elements: " \
                "two zeros and the sentence encoding."
            assert all(len(buf) == 1 for buf in self.bufs), \
                "Stacks should be fully shifted and have 1 zero."

        return [stack[-1] for stack in self.stacks], transition_acc, transition_loss


class BaseModel(nn.Module):

    optimize_transition_loss = True

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
                 encode_style=None,
                 encode_reverse=None,
                 encode_bidirectional=None,
                 encode_num_layers=None,
                 use_skips=False,
                 lateral_tracking=None,
                 use_tracking_in_composition=None,
                 use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
                 num_mlp_layers=None,
                 mlp_bn=None,
                 use_projection=None,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature
        self.hidden_dim = hidden_dim = model_dim / 2

        args = Args()
        args.lateral_tracking = lateral_tracking
        args.use_tracking_in_composition = use_tracking_in_composition
        args.size = model_dim/2
        args.tracker_size = tracking_lstm_hidden_dim
        args.transition_weight = transition_weight

        self.initial_embeddings = initial_embeddings
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        classifier_dropout_rate = 1. - classifier_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        # Build parsing component.
        self.spinn = self.build_spinn(args, vocab, use_skips)

        # Build classiifer.
        features_dim = self.get_features_dim()
        self.mlp = MLP(features_dim, mlp_dim, num_classes,
            num_mlp_layers, mlp_bn, classifier_dropout_rate)

        # The input embeddings represent the hidden and cell state, so multiply by 2.
        self.embedding_dropout_rate = 1. - embedding_keep_rate
        input_embedding_dim = args.size * 2

        # Projection will effectively be done by the encoding network.
        use_projection = True if encode_style is None else False

        # Create dynamic embedding layer.
        self.embed = Embed(input_embedding_dim, vocab.size, vectors=vocab.vectors, use_projection=use_projection)

        # Optionally build input encoder.
        if encode_style is not None:
            self.encode = self.build_input_encoder(encode_style=encode_style,
                word_embedding_dim=word_embedding_dim, model_dim=model_dim,
                num_layers=encode_num_layers, bidirectional=encode_bidirectional, reverse=encode_reverse,
                dropout=self.embedding_dropout_rate)

    def get_features_dim(self):
        features_dim = self.hidden_dim * 2 if self.use_sentence_pair else self.hidden_dim
        if self.use_sentence_pair:
            if self.use_difference_feature:
                features_dim += self.hidden_dim
            if self.use_product_feature:
                features_dim += self.hidden_dim
        return features_dim

    def build_features(self, h):
        if self.use_sentence_pair:
            h_prem, h_hyp = h
            features = [h_prem, h_hyp]
            if self.use_difference_feature:
                features.append(h_prem - h_hyp)
            if self.use_product_feature:
                features.append(h_prem * h_hyp)
            features = torch.cat(features, 1)
        else:
            features = h[0]
        return features

    def build_input_encoder(self, encode_style="LSTM", word_embedding_dim=None, model_dim=None,
                            num_layers=None, bidirectional=None, reverse=None, dropout=None):
        if encode_style == "LSTM":
            encoding_net = LSTM(word_embedding_dim, model_dim,
                num_layers=num_layers, bidirectional=bidirectional, reverse=reverse,
                dropout=dropout)
        else:
            raise NotImplementedError
        return encoding_net

    def build_spinn(self, args, vocab, use_skips):
        return SPINN(args, vocab, use_skips=use_skips)

    def build_example(self, sentences, transitions):
        raise Exception('Not implemented.')

    def spinn_hook(self, state):
        pass

    def run_spinn(self, example, use_internal_parser, validate_transitions=True):
        self.spinn.reset_state()
        state, transition_acc, transition_loss = self.spinn(example,
                               use_internal_parser=use_internal_parser,
                               validate_transitions=validate_transitions)
        self.spinn_hook(state)
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

        if hasattr(self, 'encode'):
            to_encode = torch.cat([e.unsqueeze(0) for e in embeds], 0)
            encoded = self.encode(to_encode)
            embeds = [x.squeeze() for x in torch.chunk(encoded, b, 0)]

        # Make Buffers
        embeds = [torch.chunk(x, l, 0) for x in embeds]
        buffers = [list(reversed(x)) for x in embeds]

        example.bufs = buffers

        h, transition_acc, transition_loss = self.run_spinn(example, use_internal_parser, validate_transitions)

        self.spinn_outp = h

        self.transition_acc = transition_acc
        self.transition_loss = transition_loss

        # Build features
        features = self.build_features(h)

        output = self.mlp(features)

        self.output_hook(output, sentences, transitions, y_batch)

        return output


class SentencePairModel(BaseModel):

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

    def run_spinn(self, example, use_internal_parser=False, validate_transitions=True):
        state_both, transition_acc, transition_loss = super(SentencePairModel, self).run_spinn(
            example, use_internal_parser, validate_transitions)
        batch_size = len(state_both) / 2
        h_premise = get_h(torch.cat(state_both[:batch_size], 0), self.hidden_dim)
        h_hypothesis = get_h(torch.cat(state_both[batch_size:], 0), self.hidden_dim)
        return [h_premise, h_hypothesis], transition_acc, transition_loss


class SentenceModel(BaseModel):

    def build_example(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def run_spinn(self, example, use_internal_parser=False, validate_transitions=True):
        state, transition_acc, transition_loss = super(SentenceModel, self).run_spinn(
            example, use_internal_parser, validate_transitions)
        h = get_h(torch.cat(state, 0), self.hidden_dim)
        return [h], transition_acc, transition_loss
