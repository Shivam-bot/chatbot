import json
from chatbot import tokenize_sentence, stemming_word, bag_of_word
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import NeuralNetwork


with open("intents.json", "r") as f:
    intents = json.load(f)

all_words = []
all_tags = []
tag_pattern = []

for intent in intents['intents']:
    tag = intent['tag']
    all_tags.append(tag)
    for pattern in intent['patterns']:
        tokenized_word = tokenize_sentence(pattern)
        all_words.extend(tokenized_word)
        tag_pattern.append((tokenized_word, tag))


all_words = sorted(set(stemming_word(all_words)))
all_tags = sorted(set(all_tags))

x_train = []
y_train = []


for (pattern, tag) in tag_pattern:
    word_bag = bag_of_word(pattern, all_words)
    x_train.append(word_bag)
    label = all_tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatbotDataset(Dataset):

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


batch_size = 8
num_epochs = 1000
learning_rate = 0.001

dataset = ChatbotDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

input_size, hidden_size, num_classes = len(all_words), 8, len(all_tags)


model = NeuralNetwork(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": num_classes,
    "all_words": all_words,
    "tags": all_tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')






