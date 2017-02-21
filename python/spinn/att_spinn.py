import numpy as np
from torch import nn


class SentencePairTrainer():
    """
    required by the framework, fat_classifier.py @291,295
    init as classifier_trainer at fat_classifier.py @337
    """
    pass


class AttentionModel():

    def __init__(self):
        # default define, TODO
        pass

    def soft_attention(self, stack1, ):
        """

        :param stack1:
        :param stack2:
        :return:
        """

class BaseClassifierModel(nn.module):

    def __init__(self):
    """
    """
    pass

    def forward(self):
        """

        :return: likelihood []
        """
        return [-1, -1, -1] # TODO


class SentencePairModel(BaseClassifierModel):
    """
    required by the framework, fat_classifier.py@296
    init as model at fat_classifier.py@300
    """
    def __init__(self):
        pass

    def forward(self, *input):
        pass

class SentenceModel(BaseClassifierModel):
    """
    required by the framework, fat_classifier.py@296
    init as model at fat_classifier.py@300
    because attention model take two sentences, this model might never be used
    """
    pass


