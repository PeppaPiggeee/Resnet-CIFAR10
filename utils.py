import subprocess
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import os
import glob

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.

def set_device():
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPUs available")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def load_ckpts(args):
    latest, best = [], []
    if args.resume_last:
        latest = glob.glob(os.path.join(args.ckpt_dir, "*_epoch.ckpt"))[0]
    if args.resume_best:
        best = glob.glob(os.path.join(args.ckpt_dir, "best.ckpt"))[0]
    return latest, best

def load(model, args, optimizer= None, best= False):
    latest_ckpt, best_ckpt = load_ckpts(args)
    ckpt = latest_ckpt if not best else best_ckpt
    print(f'==> Loading model from {ckpt}...')
    dict = torch.load(ckpt)
    epoch = dict['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(dict['optimizer'])
    saved_state_dict = dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print(f"{k} is not in the checkpoint")
            new_state_dict[k] = v

    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model, optimizer, epoch


def save_checkpoint(model, optimizer, epoch, args, best= False):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    latest, _ = load_ckpts(args)
    if not best and len(latest) > 0:
        for old_ckpt in latest:
            subprocess.check_call(f'rm -rf "{old_ckpt}"', shell=True)
    fn = 'best.ckpt' if best else f"{epoch}_epoch.ckpt"
    torch.save({'model': state_dict,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()},
                 os.path.join(args.ckpt_dir, fn))

def load_CIFAR(args, train= True):
    if train:
        transform= transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    try:
        ds = datasets.CIFAR10(args.data_dir, train=train, download=False, transform=transform)
    except:
        ds = datasets.CIFAR10(args.data_dir, train=train, download=True, transform=transform)
    dl = data.DataLoader(ds, args.bs)
    return dl
