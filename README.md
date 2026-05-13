# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.
Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model
<img width="1336" height="1029" alt="Screenshot 2026-05-12 103812" src="https://github.com/user-attachments/assets/c0a4d8c8-c300-4d73-8ebc-15886d211346" />


## DESIGN STEPS

STEP 1:
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

STEP 2:
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

STEP 3:
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

STEP 4:
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

STEP 5:
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

STEP 6:
Evaluate the trained model on test images and verify the classification accuracy for new unseen images.


## PROGRAM

### Name: Dinesh V

### Register Number: 212224040076

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize model, loss function, and optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)



## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    for images, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print('Name: Dinesh V ')
    print('Register Number: 212224040076  ')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


```

### OUTPUT

## Training Loss per Epoch

<img width="381" height="217" alt="image" src="https://github.com/user-attachments/assets/84c7cd33-4894-45b3-bb27-b8a124ddad12" />


## Confusion Matrix

<img width="939" height="823" alt="image" src="https://github.com/user-attachments/assets/e8b0db70-76d2-4a49-be8d-cecacdfbb5ce" />


## Classification Report

<img width="674" height="443" alt="image" src="https://github.com/user-attachments/assets/f6f9c542-6611-4931-84f3-c201c35c01d7" />


### New Sample Data Prediction

<img width="600" height="623" alt="image" src="https://github.com/user-attachments/assets/4c849262-444d-45cf-a523-18c12e2ed988" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
