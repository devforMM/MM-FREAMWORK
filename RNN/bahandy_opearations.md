📖 README: Toy RNN Encoder-Decoder with Attention (PyTorch)
This module implements a simple, educational version of an RNN-based encoder-decoder architecture with attention for sequence-to-sequence (seq2seq) tasks, using manual matrix operations with PyTorch tensors.

📦 Components:
✅ Encoder
A basic encoder class for processing input sequences.

Attributes:

layers: List of RNN layers

parameters: List of all RNN layer parameters

states: Hidden states saved at each time step during forward pass

Methods:

add_layer(vocab_size, hidden_size)
Adds an RNN layer to the encoder

train(len_seq, embeddings)
Runs a forward pass over len_seq sequence length and embeddings through each RNN layer, saving hidden states

✅ Decoder
A decoder class implementing RNN decoding with a simple additive attention mechanism.

Attributes:

loss: Cross-entropy loss function

parameters: List of parameters for each layer

optimizer: Adam optimizer placeholder (note: incomplete implementation)

w_s, w_h, v_a: Weight matrices for computing attention scores

encoder_states: Saved encoder hidden states

st: Current decoder hidden state

wy, by: Output layer parameters

Methods:

linear(hiden_state)
Projects hidden state to output logits using a linear transformation

get_encoder_sates(states)
Retrieves encoder hidden states for attention

socre()
Computes attention weights and weighted context vector using additive attention:

ini
Copier
Modifier
e_t = v_a * tanh(w_s * encoder_states + w_h * st)
a_t = softmax(e_t)
context = sum(a_t * encoder_states)
add_layer(vocab_size, hidden_size)
Adds an RNN layer to the decoder

train(len_seq, ct, seq, ytrain)
Performs training over epochs, updating hidden states, computing attention, predictions, and loss gradients manually.

📌 Notes:
This implementation is a forward-only pedagogical version; gradients and optimizer are not fully functional.

Uses additive attention (Bahdanau-style) via a simple score computation between encoder states and the decoder’s current hidden state.

Loss computed with CrossEntropyLoss

The design mimics seq2seq with attention, commonly used in machine translation or text generation.

📚 Usage Example:
python
Copier
Modifier
encoder = Encoder()
encoder.add_layer(vocab_size=30, hidden_size=64)
# process sequence embeddings and save hidden states

decoder = Decoder(attention_size=32, hidden_size=64, vocab_size=30)
decoder.get_encoder_sates(encoder.states)
# run training loop on token sequences and target labels