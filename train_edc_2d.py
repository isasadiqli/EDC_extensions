import torch
from torch.utils.data import DataLoader
from edc_2d import EDC2D
from datasets import MRNetSliceDataset


def train_edc_2d(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for slices, _ in train_loader:
        optimizer.zero_grad()
        batch_loss = 0
        for slice in slices:
            inputs = slice.to(device)
            encoded, decoded = model(inputs)
            loss = model.edc_loss(inputs, decoded)
            loss.backward()
            batch_loss += loss.item()
        optimizer.step()
        train_loss += batch_loss / len(slices)
    return train_loss / len(train_loader)


def test_edc_2d(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for slices, _ in test_loader:
            batch_loss = 0
            for slice in slices:
                inputs = slice.to(device)
                encoded, decoded = model(inputs)
                loss = model.edc_loss(inputs, decoded)
                batch_loss += loss.item()
            test_loss += batch_loss / len(slices)
    return test_loss / len(test_loader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDC2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = MRNetSliceDataset(root_dir='/path/to/mrnet', train=True)
    test_dataset = MRNetSliceDataset(root_dir='/path/to/mrnet', train=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_edc_2d(model, train_loader, optimizer, device)
        test_loss = test_edc_2d(model, test_loader, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
