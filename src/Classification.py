from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

from compute_mean_std import compute_mean_std, read_mean_std

from tqdm.auto import tqdm

_stats_transform = transforms.Compose([transforms.ToTensor()])
_stats_dataset = ImageFolder(root="./transformations", transform=_stats_transform)
_stats_loader = DataLoader(dataset=_stats_dataset, batch_size=1, shuffle=False, num_workers=0)

compute_mean_std(_stats_loader)
mean, std = read_mean_std('./models/mean_std.csv')
print(f"the mean of the dataset is {mean}, and the std is {std}")

# Define transformations for the training and test sets
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

batch_size = 1
number_of_labels = 4
size_test_set = 200

# Create a dataset for training and a dataset for testing
dataset = ImageFolder(root="./transformations", transform=transformations)

# train_path = "./dataset_split/train"
# test_path = "./dataset_split/val"

# delete_augmented_images(train_path)
# print("Running data augmentation on training set...")
# augmentation(train_path)
# print("Data augmentation done.\n")

train_set, test_set = random_split(dataset, [len(dataset) - size_test_set, size_test_set])
# train_set = ImageFolder(root=train_path, transform=transformations)
# test_set = ImageFolder(root=test_path, transform=transformations)

# Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of images in a training set is: ", len(train_loader)*batch_size)

# Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ('Black Rot', 'Healthy', 'Rust', 'Scab')

# Instantiate a neural network model 
from Network import Network
model = Network(batch_size)

from torch.optim import Adam
import torch.nn as nn
import torch
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

from torch.autograd import Variable

# Function to save the model
def saveModel():
    path = "./models/myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
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
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

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

        # Compute and print the average accuracy fo this epoch when tested over all test images
        accuracy = testAccuracy()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader) * batch_size))
        print('For epoch', epoch+1, 'the test accuracy over the whole test set is %d %%\n' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy
    

if __name__ == "__main__":
    
    # Let's build our model
    train(num_epochs=3)
    print('Finished Training')

    # Test which classes performed well
    testAccuracy()
