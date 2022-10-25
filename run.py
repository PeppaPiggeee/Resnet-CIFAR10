import torch
import torch.nn as nn
from utils import *
from model import Resnet
import argparse

def run(args):
    device = set_device()
    set_seed(args)

    print('==> Preparing CIFAR1O...')
    train_dl, test_dl = load_CIFAR(args)

    print('==> Building 34-layer Resnet...')
    model = Resnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    print('==> Training 34-layer Resnet ...')
    train_loss_, test_loss_ = [], []
    train_acc_, test_acc_ = [], []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_dl, optimizer, loss_func, device)
        test_loss, test_acc = test_epoch(model, test_dl, loss_func, device)

        train_loss_.append(train_loss)
        test_loss_.append(test_loss)
        train_acc_.append(train_loss)
        test_acc_.append(test_acc)

        if epoch % 10 == 0:
            print(f"epoch {epoch}: train loss: {train_loss}, train accuracy {train_acc}, "
                  f"test loss: {test_loss}, test accuracy {test_acc}")

    print('==> Saving results in result.png ...')
    plot(train_loss_, test_loss_, train_acc_, test_acc_)


def train_epoch(model, data, optimizer, loss_func, device):
    model.train()
    l_sum, correct, n = 0.0, 0, 0
    for x, y in data:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_func(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l_sum += loss.item()
        prediction = torch.argmax(y_pred, -1)
        correct += (prediction == y).sum().item()
        n += x.shape[0]

    acc = torch.true_divide(correct, n)
    return l_sum, acc

def test_epoch(model, data, loss_func, device):
    l_sum, correct, n = 0.0, 0, 0
    model.eval()
    for x, y in data:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_func(y_pred, y)

        l_sum += loss.item()
        prediction = torch.argmax(y_pred, -1)
        correct += (prediction == y).sum().item()
        n += x.shape[0]

    acc = torch.true_divide(correct, n)
    return l_sum, acc

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--bs", type=int, default=64, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=150, help="epochs", required=False)
    PARSER.add_argument("--lr", type=float, default=0.001, help="learning_rate", required=False)
    PARSER.add_argument("--data_dir", type=str, default="data", help="CIFAR_root_dir", required=False)
    PARSER.add_argument('--seed', type=int, default=1234)
    args, _ = PARSER.parse_known_args()

    run(args)












