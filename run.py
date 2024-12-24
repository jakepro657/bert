from bert import Bert
import torch
from torch.optim import Adam
import torch.nn as nn

def train(model, data_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs, pooled_output = model(inputs)
            loss = criterion(pooled_output, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Example usage
if __name__ == "__main__":
    # Dummy data loader

    data_loader = [(
        torch.rand(32, 10, 512),  # batch_size x seq_len x d_model
        torch.randint(0, 2, (32, 512)).float()  # batch_size x d_model
    ) for _ in range(100)]

    model = Bert()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train(model, data_loader, criterion, optimizer)