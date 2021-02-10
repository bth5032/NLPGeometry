# External imports
import torch
import torch.nn as nn

# Internal imports
from src.models.layers import SO3Block, ClassificationHead, ConvNet


class SO3Classifier(nn.Module):
    def __init__(self):
        super(SO3Classifier, self).__init__()
        self.euclidean_network = ConvNet()
        self.lie_block = SO3Block(self.euclidean_network)
        self.head = ClassificationHead()

        # Lie block param: single global context, for now.
        self.register_buffer("initial_context", torch.tensor([1, 0, 0], dtype=torch.float))

    def forward(self, x):
        context = self.lie_block(x, self.initial_context)
        output = self.head(context).reshape(1, -1)
        return output
