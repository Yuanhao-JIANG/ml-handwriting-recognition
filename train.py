from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import LeNet5


# get data
dataset = datasets.MNIST("./data", True, transforms.ToTensor(), None, True)

# config info
save_dir = Path("./saved_model").absolute()
save_dir.mkdir(parents=True, exist_ok=True)
train_len = int(len(dataset)*0.9)
val_len = len(dataset) - train_len
batch_size = 50
epoch_size = 50
class_num = 10
early_stop_cnt = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

# prepare data
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# load model
net = LeNet5(class_num)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
net = net.to(device)


# train and validation
def val(val_set, model):
    model.eval()                             # set model to evalutation mode
    total_loss = 0
    for x, y in val_set:                     # iterate through the dataloader
        x, y = x.to(device), y.to(device)    # move data to device (cpu/cuda)
        with torch.no_grad():                # disable gradient calculation
            logits, probs = model(x)         # forward pass (compute output)
            val_loss = criterion(logits, y)  # compute loss
        total_loss += val_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(val_set.dataset)             # compute averaged loss

    return total_loss


def train():
    min_loss = 100
    current_stop_cnt = 0
    current_epoch = 0

    for epoch in range(epoch_size):
        net.train()
        for imgs, labels_int in train_loader:
            labels = torch.zeros([batch_size, class_num])
            for i in range(batch_size):
                labels[i][labels_int[i]] = 1

            imgs, labels = imgs.to(device), labels.to(device)
            logits, probs = net(imgs)

            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_loss = val(val_loader, net)
        if val_loss < min_loss:
            min_loss = val_loss
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, min_loss))
            torch.save(net.state_dict(), save_dir / f"model.pt")
            current_stop_cnt = 0
        else:
            current_stop_cnt += 1

        current_epoch = epoch + 1
        if current_stop_cnt >= early_stop_cnt:
            break

    print('Finished training after {} epochs'.format(current_epoch))
