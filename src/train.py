from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

from utils.utils import compute_mean_std, read_mean_std
from tqdm.auto import tqdm
import argparse

from classes.Network import Network
from torch.optim import AdamW
import torch.nn as nn
import torch
from torch.autograd import Variable
import os
import zipfile


batch_size = 16
number_of_labels = 4
size_test_set = 200


def prepare_datasets(training_dir: str, checkpoints_dir: str, fruit: str):
    _stats_transform = transforms.Compose([transforms.ToTensor()])
    _stats_dataset = ImageFolder(root=training_dir, transform=_stats_transform)
    _stats_loader = DataLoader(dataset=_stats_dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0)

    compute_mean_std(_stats_loader, fruit)
    mean, std = read_mean_std(os.path.join(checkpoints_dir,
                                           'mean_std' + fruit + '.csv'))
    print(f"the mean of the dataset is {mean}, and the std is {std}")

    # Define transformations for the training and test sets
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create a dataset for training and a dataset for testing
    dataset = ImageFolder(root=training_dir, transform=transformations)

    dataset_dim = [len(dataset) - size_test_set, size_test_set]
    train_set, test_set = random_split(dataset, dataset_dim)
    # train_set = ImageFolder(root=train_path, transform=transformations)
    # test_set = ImageFolder(root=test_path, transform=transformations)

    # Create a loader for the training
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    print("The number of images in a training set is: ",
          len(train_loader)*batch_size)

    # Create a loader for the testing
    # Note that each shuffle is set to false for the test loader.
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
    print("The number of images in a test set is: ",
          len(test_loader)*batch_size)
    print("The number of batches per epoch is: ",
          len(train_loader))

    return train_loader, test_loader


# Function to test the model with the test dataset
def test_accuracy(model: Network, test_loader: DataLoader):

    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images.to(device))
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy


# Training function.
def train(model: Network,
          num_epochs: int,
          train_loader: DataLoader,
          test_loader: DataLoader,
          checkpoint_dir: str,
          fruit: str):

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )

    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, (images, labels) in tqdm(enumerate(train_loader, 0),
                                        total=len(train_loader),
                                        desc=f"Epoch {epoch+1}/{num_epochs}"):

            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value

        # Compute and print the average accuracy for this epoch
        accuracy = test_accuracy(model=model, test_loader=test_loader)
        scheduler.step(accuracy)
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
        print('[%d] loss: %.3f' %
              (epoch + 1, running_loss / len(train_loader) * batch_size))
        print('For epoch',
              epoch+1,
              'the test accuracy over the whole test set is %d %%\n' %
              (accuracy))

        torch.save(model.state_dict(),
                   os.path.join(checkpoint_dir, "lastModel" + fruit + ".pth"))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir,
                                    "bestModel" + fruit + ".pth"))


def store_model_compressed(src_path: str, model_path: str):
    with zipfile.ZipFile('./test.zip', 'w') as zipf:
        for dirpath, dirnames, filenames in os.walk(src_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                archive_file_path = os.path.relpath(file_path, src_path)
                zipf.write(file_path,
                           os.path.join(src_path, archive_file_path))
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                archive_file_path = os.path.relpath(file_path, model_path)
                zipf.write(file_path,
                           os.path.join(model_path, archive_file_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--src",
        type=str,
        default="./Apple",
        help="The folder you want to train on."
    )

    parser.add_argument(
        "-m", "--models-dir",
        type=str,
        default="./models",
        help="The folder you want to save the models."
    )

    parser.add_argument(
        "-f", "--fruit",
        type=str,
        choices={"Apple", "Grape"},
        default="Apple",
        help="The fruit leaf you want to predict"
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Let's build our model
    model = Network()
    train_loader, test_loader = prepare_datasets(args.src,
                                                 args.models_dir,
                                                 args.fruit)
    train(model=model,
          num_epochs=20,
          train_loader=train_loader,
          test_loader=test_loader,
          checkpoint_dir=args.models_dir,
          fruit=args.fruit,)
    print('Finished Training')

    # Test which classes performed well
    test_accuracy(model=model, test_loader=test_loader)

    # Store the model and augmented dataset
    store_model_compressed(args.src, args.models_dir)
