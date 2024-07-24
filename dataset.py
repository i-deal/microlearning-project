import torch
import torchvision

def load_dataset(dataset, batch_size):
    if dataset == 'MNIST':
        mnist_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # Convert images to tensor
            torchvision.transforms.Normalize((0.1307,), (0.3081,))  # Normalize the images with mean and standard deviation
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transforms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # Enable persistent_workers=True if more than 1 worker to save CPU
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transforms)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)   # Enable persistent_workers=True if more than 1 worker to save CPU

    elif 'CIFAR10': # CIFAR-10
        # Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # Enable persistent_workers=True if more than 1 worker to save CPU
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)   # Enable persistent_workers=True if more than 1 worker to save CPU

    return (trainset, testset), (trainloader, testloader)