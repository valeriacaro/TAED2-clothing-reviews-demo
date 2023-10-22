"""
This module is created to contain the classes
related to the first deep learning model we tried
and all functions related to it.
"""

import torch

class Dictionary:
    """
    This class constructs a dictionary to map
    the tokens that encode the words to an index,
    and viceversa.
    """
    def __init__(self):
        """
            Declare the initialization of the dictionary.
        """
        self.token2idx = {}
        self.idx2token = []

    def add_token(self, token):
        """
            Add a token to the dictionary and return its index.

            Args:
                token (str):
                - The token to be added to the dictionary.

            Returns:
                int: The index of the added token.
        """
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def __len__(self):
        """
            Get the length of the dictionary.

            Returns:
                int: The number of tokens in the dictionary.
        """
        return len(self.idx2token)


class CharRNNClassifier(torch.nn.Module):
    """
    This class defines the RNN Classifier
    itself.
    """
    def __init__(
            self, input_size,
            embedding_size,
            hidden_size,
            output_size,
            model="lstm",
            num_layers=4,
            bidirectional=False,
            pad_idx=0
    ):
        """
               Initialize the Character RNN Classifier model.

               Args:
                   input_size (int): The size of the input vocabulary.
                   embedding_size (int): The size of word embeddings.
                   hidden_size (int): The size of the hidden layers.
                   output_size (int): The size of the output (number of classes).
                   model (str): The RNN model type ("lstm" or "gru").
                   num_layers (int): The number of RNN layers.
                   bidirectional (bool): Whether to use bidirectional RNN.
                   pad_idx (int): The padding index for embeddings.

               Returns:
                   None
        """
        super().__init__()
        self.model = model.lower()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embed = torch.nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)
        if self.model == "gru":
            self.rnn = torch.nn.GRU(
                embedding_size, hidden_size, num_layers, bidirectional=bidirectional
            )
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(
                embedding_size, 2*hidden_size, num_layers, bidirectional=bidirectional
            )
        self.h2o = torch.nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = torch.nn.Dropout(0.2, inplace=True)


    def forward(self, user_input, input_lengths):
        """
                Forward pass of the model.

                Args:
                    user_input (Tensor): Input sequences.
                    input_lengths (Tensor): Lengths of input sequences.

                Returns:
                    Tensor: Model output.
        """
        # T x B
        encoded = self.embed(user_input)
        # T x B x E
        packed = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_lengths)
        # Packed T x B x E
        output, _ = self.rnn(packed)
        # Packed T x B x H
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, padding_value=float('-inf')
        )
        # T x B x H
        output, _ = padded.max(dim=0)
        # Dropout
        output = self.dropout(output)
        # B x H
        output = self.h2o(output.view(-1, self.num_directions * self.hidden_size))
        # B x O
        return output


if __name__ == '__main__':

    pass
