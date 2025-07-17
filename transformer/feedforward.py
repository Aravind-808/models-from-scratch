import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    '''
    Feed Forward Network used in Transformers:
    - Applies two linear layers with ReLU in between
    - Operates independently on each position
    '''
    def __init__(self, d_model, d_ff):
        '''
        d_model: input/output dimension
        d_ff: hidden layer dimension (usually larger)
        '''
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff) # first linear layer
        self.fc2 = nn.Linear(d_ff, d_model) # second linear layer
        self.relu = nn.ReLU() # non-linearity

    def forward(self, x):
        '''
        x: input tensor of shape (batch, seq_len, d_model)
        '''
        return self.fc2(self.relu(self.fc1(x))) # FFN: Linear → ReLU → Linear
