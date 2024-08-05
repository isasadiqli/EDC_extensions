import torch
from torch.utils.data import DataLoader
from extentions.edc_3d import EDC3D
from datasets import MRNetDataset


def train_edc_3d(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        inputs = data.to(device)
        optimizer.zero_grad()
        encoded, decoded = model(inputs)
        loss = model.edc_loss(inputs, decoded)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


def test_edc_3d(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            inputs = data.to(device)
            encoded, decoded = model(inputs)
            loss = model.edc_loss(inputs, decoded)
            test_loss += loss.item()
    return test_loss / len(test_loader)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDC3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = MRNetDataset(root_dir='/mrnet', train=True)
    test_dataset = MRNetDataset(root_dir='/mrnet', train=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_edc_3d(model, train_loader, optimizer, device)
        test_loss = test_edc_3d(model, test_loader, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
