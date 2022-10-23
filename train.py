import torch
import torch.nn as nn
from utils import *
from model import Resnet
import argparse

def run(args):
    train(args) if not args.test else test(args)

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
    return l_sum, correct, n

def train(args):
    device = set_device()
    set_seed(args)

    print('==> Preparing CIFAR1O...')
    train_dl = load_CIFAR(args)

    print('==> Building 34-layer Resnet...')
    model = Resnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    try:
        _, _, epoch_ = load(model, args, optimizer= optimizer)
    except:
        epoch_ = 1

    print('==> Training 34-layer Resnet ...')
    best_acc = 0.0
    for epoch in range(epoch_, args.epochs + 1):
        loss, correct, n = train_epoch(model, train_dl, optimizer, loss_func, device)
        print(f"==> Saving model and optimizer state at epoch {epoch}...")
        save_checkpoint(model, optimizer, epoch, args)

        acc = torch.true_divide(correct, n)
        if epoch % 1000 == 0:
            print(f"epoch {epoch}: loss {loss}, accuracy {acc}")
        if acc > best_acc:
            best_acc = acc
            if args.resume_best:
                print(f"==> Saving best model and optimizer state at epoch {epoch}...")
                save_checkpoint(model, optimizer, epoch, args, best= True)


def test(args):
    device = set_device()

    print('==> Preparing CIFAR1O...')
    test_dl = load_CIFAR(args, train= False)

    print('==> Building 34-layer Resnet...')
    model = Resnet().to(device)

    try:
        _, _, epoch_ = load(model, args)
    except:
        print('Warning! Model checkpoints not found!')

    print('==> Test 34-layer Resnet ...')
    correct, n = 0, 0
    model.eval()
    for x, y in test_dl:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)

        prediction = torch.argmax(y_pred, -1)
        correct += (prediction == y).sum().item()
        n += x.shape[0]
    print(f'test acc: {100 * torch.true_divide(correct, n)}%')




if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--bs", type=int, default=64, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=150, help="epochs", required=False)
    PARSER.add_argument("--lr", type=float, default=0.001, help="learning_rate", required=False)
    PARSER.add_argument("--data_dir", type=str, default="data", help="CIFAR_root_dir", required=False)
    PARSER.add_argument("--ckpt_dir", type=str, default="checkpoints", help="CIFAR_root_dir", required=False)
    PARSER.add_argument('--seed', type=int, default=1234)
    PARSER.add_argument('--resume_last', action='store_true')
    PARSER.add_argument('--resume_best', action='store_true')
    PARSER.add_argument('--test', action='store_true')
    args, _ = PARSER.parse_known_args()

    run(args)












