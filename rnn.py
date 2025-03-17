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
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        out, hidden = self.rnn(inputs)
        # [to fill] obtain output layer representations
        z = self.W(out)
        # [to fill] sum over output
        z_sum = torch.sum(z, dim=0)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(z_sum)
        return predicted_vector


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

    # Fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    
    # Load test data
    with open(args.test_data) as test_f:
        test_json = json.load(test_f)
    test_data = []
    for elt in test_json:
        test_data.append((elt["text"].split(), int(elt["stars"]-1)))
    
    print("========== Vectorizing data ==========")
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))
    
    # Define hidden dimensions to test
    hidden_dims = [50, 100, 150, 200, 250]
    epochs = 5
    
    # Dictionary to store results
    results = {hd: {"train_acc": [], "val_acc": [], "test_acc": None} for hd in hidden_dims}
    
    for hd in hidden_dims:
        print(f"\n========== Training with hidden dimension {hd} ==========")
        model = RNN(50, hd)  # 50 is the input dimension from word embeddings
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            correct = 0
            total = 0
            loss_total = 0
            loss_count = 0
            start_time = time.time()
            print(f"Training started for epoch {epoch + 1}")
            random.shuffle(train_data)
            minibatch_size = 16
            N = len(train_data)
            
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    input_words = " ".join(input_words)

                    # Remove punctuation
                    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                    # Look up word embedding dictionary
                    vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

                    # Transform the input into required shape
                    vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                    output = model(vectors)

                    # Get loss
                    example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                    # Get predicted label
                    predicted_label = torch.argmax(output)

                    correct += int(predicted_label == gold_label)
                    total += 1
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss

                loss = loss / minibatch_size
                loss_total += loss.data
                loss_count += 1
                loss.backward()
                optimizer.step()
                
            train_accuracy = correct / total
            results[hd]["train_acc"].append(train_accuracy)
            print(f"Average loss: {loss_total/loss_count}")
            print(f"Training completed for epoch {epoch + 1}")
            print(f"Training accuracy for epoch {epoch + 1}: {train_accuracy}")
            print(f"Training time for this epoch: {time.time() - start_time}")

            # Validation phase
            model.eval()
            correct = 0
            total = 0
            start_time = time.time()
            print(f"Validation started for epoch {epoch + 1}")
            
            with torch.no_grad():
                for input_words, gold_label in tqdm(valid_data):
                    input_words = " ".join(input_words)
                    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                    vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

                    vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                    output = model(vectors)
                    predicted_label = torch.argmax(output)
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
            for input_words, gold_label in tqdm(test_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
        
        test_accuracy = correct / total
        results[hd]["test_acc"] = test_accuracy
        print(f"Test accuracy for hidden dimension {hd}: {test_accuracy}")
    
    # Write results to file
    print("\n========== Summary of Results ==========")
    with open("rnn_results.txt", "w") as f:
        f.write("RNN Hidden Dimension Results Summary\n")
        f.write("==================================\n\n")
        
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
    
    print(f"Results written to rnn_results.txt")
