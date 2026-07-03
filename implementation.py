import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleFusionModel(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim):
        super(SimpleFusionModel, self).__init__()
        self.feature_extractor1 = nn.Sequential(
            nn.Linear(input_dim1, hidden_dim),
            nn.ReLU()
        )
        self.feature_extractor2 = nn.Sequential(
            nn.Linear(input_dim2, hidden_dim),
            nn.ReLU()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, input1, input2):
        features1 = self.feature_extractor1(input1)
        features2 = self.feature_extractor2(input2)
        fused_features = torch.cat((features1, features2), dim=1)
        output = self.fusion_layer(fused_features)
        return output

def generate_dummy_data(num_samples, input_dim1, input_dim2, num_classes):
    data1 = np.random.rand(num_samples, input_dim1).astype(np.float32)
    data2 = np.random.rand(num_samples, input_dim2).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_samples,))
    return data1, data2, labels

def train_model(model, dataloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for input1, input2, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input1, input2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    # Parameters
    input_dim1 = 10
    input_dim2 = 8
    hidden_dim = 16
    output_dim = 3
    num_samples = 1000
    batch_size = 32
    learning_rate = 0.001
    epochs = 10

    # Generate dummy data
    data1, data2, labels = generate_dummy_data(num_samples, input_dim1, input_dim2, output_dim)
    dataset = TensorDataset(torch.tensor(data1), torch.tensor(data2), torch.tensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, and optimizer
    model = SimpleFusionModel(input_dim1, input_dim2, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, epochs)

    # Test the model with dummy data
    test_data1, test_data2, test_labels = generate_dummy_data(10, input_dim1, input_dim2, output_dim)
    test_input1 = torch.tensor(test_data1)
    test_input2 = torch.tensor(test_data2)
    predictions = model(test_input1, test_input2)
    print("Predictions:", predictions.argmax(dim=1).numpy())
    print("True Labels:", test_labels)