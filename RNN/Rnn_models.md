📖 README: Simple RNN, LSTM, and GRU Implementations in PyTorch
This project provides basic implementations of:

Vanilla RNN

LSTM (Long Short-Term Memory)

GRU (Gated Recurrent Unit)
Each class is implemented using manual matrix multiplications with random initializations via PyTorch tensors, without using torch.nn.Module or autograd.

📦 Components:
✅ Rnn
A simple vanilla recurrent neural network.

Attributes:

wh: Hidden-to-hidden weight matrix

wx: Input-to-hidden weight matrix

wy: Hidden-to-output weight matrix

by: Output bias

bh: Hidden bias

Methods:

hiden_satate(ht_1, xt)
Computes the next hidden state given previous state ht_1 and input xt

output(hiden_state)
Computes output logits from hidden state

✅ LStm
An LSTM implementation, extending Rnn.

Additional Weights:

Separate weight matrices and biases for:

Input gate

Forget gate

Output gate

Candidate cell state

Methods:

input_gate(xt, ht_1)
Computes input gate activation

forget_gate(xt, ht_1)
Computes forget gate activation

output_gate(xt, ht_1)
Computes output gate activation

candidate_info(xt, ht_1)
Computes candidate cell state value

memory_update(ft, it, ct_bar, ct_1)
Updates cell state using forget, input gates and candidate

hidden_state(ot, ct)
Computes the hidden state from the output gate and updated cell state

✅ Gru
A GRU implementation, extending Rnn.

Additional Weights:

Separate weight matrices and biases for:

Reset gate

Update gate

Candidate state

Methods:

reset_gate(xt, ht_1)
Computes reset gate activation

update(xt, ht_1)
Computes update gate activation

candidate_state(xt, rt, ht_1)
Computes candidate hidden state using reset gate

new_state(zt, ht_1, candidate)
Combines previous hidden state and candidate state based on update gate

📌 Notes:
All weights and biases are randomly initialized with torch.randn

Multiplications use * (element-wise) and @ (matrix product) explicitly

No automatic gradient tracking or optimizers are included — this is a forward-only, educational implementation

Useful for understanding the internal computations of RNNs, LSTMs and GRUs at a low level

📚 Usage Example:
python
Copier
Modifier
model = Rnn(vocab_size=10, hidden_size=20)
ht = torch.randn(1, 20)
xt = torch.randn(1, 10)
next_ht = model.hiden_satate(ht, xt)
output = model.output(next_ht)
📚 Reference:
Inspired by:

Recurrent Neural Network (RNN)

Long Short-Term Memory (LSTM)

Gated Recurrent Unit (GRU)
as introduced in classical sequence modeling and NLP literature.