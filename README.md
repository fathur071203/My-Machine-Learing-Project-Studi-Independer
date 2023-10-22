# My-Machine-Learing-Project-Studi-Independer
## MNIST Handwriting Classification using PyTorch

This repository contains the code for performing MNIST Handwritten Digit Classification using a basic Neural Network architecture implemented in PyTorch.

### Overview

The goal of this project is to explore an end-to-end workflow for solving the MNIST Handwritten Digit Classification task using a simple Neural Network architecture. This project does not utilize Convolutional Neural Networks (CNNs), making it a basic introduction to image classification with deep learning.

### Code Structure

The repository contains the following key sections:

1. **Data Loading:**

   The MNIST dataset is loaded using PyTorch's `torchvision` library. The data is normalized and split into training and test sets.

   ```python
   # Step 1: Load MNIST dataset for the data loader
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

   mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)

   mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
   test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=False)
   ```

2. **Visualization:**

   You can visualize some sample images from the dataset using the following code.

   ```python
   import matplotlib.pyplot as plt

   def plot_images(images):
     fig, axs = plt.subplots(1, len(images), figsize=(12, 6))
     for i, image in enumerate(images):
       axs[i].imshow(image.squeeze().numpy(), cmap='gray')
       axs[i].axis('off')

   images, labels = next(iter(train_loader))
   plot_images(images[:10])
   ```

3. **Neural Network Model:**

   The neural network model is implemented using PyTorch's `nn.Module`. It's a simple feedforward neural network with two hidden layers.

   ```python
   class Net(nn.Module):
     def __init__(self):
       super(Net, self).__init__()
       self.linear1 = nn.Linear(28*28, 100)
       self.linear2 = nn.Linear(100, 50)
       self.final = nn.Linear(50, 10)
       self.relu = nn.ReLU()

     def forward(self, img):
       x = img.view(-1, 28*28)
       x = self.relu(self.linear1(x))
       x = self.relu(self.linear2(x))
       x = self.final(x)
       return x

   net = Net()
   ```

4. **Hyperparameters and Training:**

   Set up hyperparameters such as the loss function, optimizer, learning rate, and training loop. An example training loop is included.

   ```python
   cross_el = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
   epoch = 20

   for epoch in range(epoch):
     net.train()
     running_loss = 0.0

     for data in train_loader:
       x, y = data
       optimizer.zero_grad()
       output = net(x.view(-1, 28*28))
       loss = cross_el(output, y)
       loss.backward()
       optimizer.step()
       running_loss += loss.item()

     print(f'[{epoch + 1}, {epoch + 1:5d}] loss : {running_loss / 2000:.5f}')
   ```

5. **Model Evaluation:**

   Evaluate the model using metrics like accuracy, confusion matrix, and classification report. The code also prints the results.

   ```python
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

   correct = 0
   total = 0
   all_labels = []
   all_predictions = []

   with torch.no_grad():
       for data in test_loader:
           images, labels = data
           outputs = net(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
           all_labels.extend(labels.numpy())
           all_predictions.extend(predicted.numpy())

   accuracy = accuracy_score(all_labels, all_predictions)
   confusion = confusion_matrix(all_labels, all_predictions)
   classification_report_str = classification_report(all_labels, all_predictions)

   print(f"Accuracy: {accuracy * 100}%")
   print("Confusion Matrix:")
   print(confusion)
   print("Classification Report:")
   print(classification_report_str)
   ```

### Conclusion

The model achieved an accuracy of approximately 97.48%, which indicates its high capability for handwritten digit classification. This result is quite impressive for a basic neural network model without using convolutional neural networks.

Feel free to use and modify this code for your MNIST digit classification tasks or as a starting point for more complex image classification projects. Enjoy exploring the world of deep learning!
