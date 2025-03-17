import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden = self.W1(input_vector)
        hidden = self.activation(hidden)

        # [to fill] obtain output layer representation
        output = self.W2(hidden)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)

        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", required=True, help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    
    # Load test data
    with open(args.test_data) as test_f:
        test_json = json.load(test_f)
    test_data = []
    for elt in test_json:
        test_data.append((elt["text"].split(), int(elt["stars"]-1)))
    
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    test_data = convert_to_vector_representation(test_data, word2index)
    
    # Define hidden dimensions to test
    hidden_dims = [50, 100, 150, 200, 250]
    epochs = 5
    
    # Dictionary to store results
    results = {hd: {"train_acc": [], "val_acc": [], "test_acc": None} for hd in hidden_dims}
    
    for hd in hidden_dims:
        print(f"\n========== Training with hidden dimension {hd} ==========")
        model = FFNN(input_dim=len(vocab), h=hd)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            correct = 0
            total = 0
            start_time = time.time()
            print(f"Training started for epoch {epoch + 1}")
            random.shuffle(train_data)
            minibatch_size = 16 
            N = len(train_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
                loss.backward()
                optimizer.step()
            
            train_accuracy = correct / total
            results[hd]["train_acc"].append(train_accuracy)
            print(f"Training completed for epoch {epoch + 1}")
            print(f"Training accuracy for epoch {epoch + 1}: {train_accuracy}")
            print(f"Training time for this epoch: {time.time() - start_time}")

            # Validation phase
            model.eval()
            correct = 0
            total = 0
            start_time = time.time()
            print(f"Validation started for epoch {epoch + 1}")
            minibatch_size = 16 
            N = len(valid_data) 
            with torch.no_grad():
                for minibatch_index in tqdm(range(N // minibatch_size)):
                    for example_index in range(minibatch_size):
                        input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                        predicted_vector = model(input_vector)
                        predicted_label = torch.argmax(predicted_vector)
                        correct += int(predicted_label == gold_label)
                        total += 1
            
            val_accuracy = correct / total
            results[hd]["val_acc"].append(val_accuracy)
            print(f"Validation completed for epoch {epoch + 1}")
            print(f"Validation accuracy for epoch {epoch + 1}: {val_accuracy}")
            print(f"Validation time for this epoch: {time.time() - start_time}")
        
        # Test phase after all epochs
        model.eval()
        correct = 0
        total = 0
        print(f"Testing model with hidden dimension {hd}")
        with torch.no_grad():
            for input_vector, gold_label in test_data:
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
        
        test_accuracy = correct / total
        results[hd]["test_acc"] = test_accuracy
        print(f"Test accuracy for hidden dimension {hd}: {test_accuracy}")
    
    # Write results to file
    print("\n========== Summary of Results ==========")
    with open("results.txt", "w") as f:
        f.write("Hidden Dimension Results Summary\n")
        f.write("================================\n\n")
        
        for hd in hidden_dims:
            f.write(f"Hidden Dimension: {hd}\n")
            f.write("-----------------\n")
            
            f.write("Training Accuracy by Epoch:\n")
            for i, acc in enumerate(results[hd]["train_acc"]):
                f.write(f"  Epoch {i+1}: {acc:.4f}\n")
            
            f.write("\nValidation Accuracy by Epoch:\n")
            for i, acc in enumerate(results[hd]["val_acc"]):
                f.write(f"  Epoch {i+1}: {acc:.4f}\n")
            
            f.write(f"\nTest Accuracy: {results[hd]['test_acc']:.4f}\n\n")
        
        # Find best model based on validation accuracy
        best_hd = max(hidden_dims, key=lambda hd: results[hd]["val_acc"][-1])
        f.write(f"Best model based on final validation accuracy: Hidden Dimension = {best_hd}\n")
        f.write(f"  Final validation accuracy: {results[best_hd]['val_acc'][-1]:.4f}\n")
        f.write(f"  Test accuracy: {results[best_hd]['test_acc']:.4f}\n")
    
    print(f"Results written to results.txt")
    