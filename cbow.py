import streamlit as st
import numpy as np

# Step 1: Define Streamlit UI Components
st.title("CBOW Model Implementation")

# Input Text
text = st.text_area("Input Text", "We are learning about natural language processing using CBOW model")

# Select the word to predict
words = text.lower().split()
selected_word = st.selectbox("Select the word to predict", words)

context_size = st.slider("Context Size", 1, 3, 1)
embedding_dim = st.slider("Embedding Dimension", 5, 50, 10)
learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
epochs = st.slider("Epochs", 1000, 10000, 10000, step=1000)

# Global variables for training data and model weights
data = []
W1 = None
W2 = None
vocab = None
word_to_ix = None
ix_to_word = None

if st.button("Train Model"):
    vocab = set(words)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}

    # Generate training data (context, target)
    data = []
    for i in range(context_size, len(words) - context_size):
        context = words[i-context_size:i] + words[i+1:i+context_size+1]
        target = words[i]
        data.append((context, target))

    # Initialize the Model Parameters
    W1 = np.random.rand(len(vocab), embedding_dim)  # Input to hidden layer weights
    W2 = np.random.rand(embedding_dim, len(vocab))  # Hidden to output layer weights

    # Define Helper Functions
    def one_hot_encoding(word, vocab_size):
        one_hot = np.zeros(vocab_size)
        one_hot[word_to_ix[word]] = 1
        return one_hot

    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

    def forward_pass(context_words):
        h = np.zeros(embedding_dim)
        for word in context_words:
            x = one_hot_encoding(word, len(vocab))
            h += np.dot(x, W1)
        h /= len(context_words)
        u = np.dot(h, W2)
        y_pred = softmax(u)
        return y_pred, h

    def cross_entropy_loss(y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred))

    def backward_pass(context_words, h, y_pred, y_true):
        dW2 = np.outer(h, (y_pred - y_true))
        dW1 = np.zeros(W1.shape)
        for word in context_words:
            x = one_hot_encoding(word, len(vocab))
            dW1 += np.outer(x, np.dot((y_pred - y_true), W2.T))
        dW1 /= len(context_words)
        return dW1, dW2

    # Train the Model
    for epoch in range(epochs):
        total_loss = 0
        for context_words, target_word in data:
            y_pred, h = forward_pass(context_words)
            y_true = one_hot_encoding(target_word, len(vocab))
            loss = cross_entropy_loss(y_pred, y_true)
            total_loss += loss
            dW1, dW2 = backward_pass(context_words, h, y_pred, y_true)
            W1 -= learning_rate * dW1
            W2 -= learning_rate * dW2
        if epoch % 100 == 0:
            print("Epoch:", epoch, "Loss:", total_loss)

    st.write("Training Complete!")

    # Test the Model
    test_index = words.index(selected_word)
    test_context = words[test_index-context_size:test_index] + words[test_index+1:test_index+context_size+1]
    if len(test_context) == context_size * 2:
        y_pred, _ = forward_pass(test_context)
        predicted_word = ix_to_word[np.argmax(y_pred)]
        st.write(f"Predicted word for context {test_context} is '{predicted_word}'")
    else:
        st.write("Not enough context words for the selected word.")
