# Developing a Neural Network Regression Model

## Name : Kevin P
## Register Number : 212224040159

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model

<img width="782" height="529" alt="image" src="https://github.com/user-attachments/assets/ddeb7400-6ef2-43da-ab53-4c913a5fed71" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM
```python

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dataset = pd.read_excel('/content/dl 1.xlsx')

# The dataset columns have trailing spaces: 'Input ' and 'Output '
X = dataset[['Input ']].values
y = dataset[['Output ']].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

def train_model(model, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        model.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")

loss_df = pd.DataFrame(ai_brain.history)
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_new = torch.tensor([[9]], dtype=torch.float32)
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)
prediction = ai_brain(X_new_tensor).item()
print(f"Prediction: {prediction}")

```





### Dataset Information

<img width="167" height="360" alt="image" src="https://github.com/user-attachments/assets/6fedf826-70fb-4c9a-972d-244444f3dbf6" />

### OUTPUT

<img width="375" height="247" alt="image" src="https://github.com/user-attachments/assets/4dee877e-e84f-4981-87c6-adb8abb063b0" />

### Training Loss Vs Iteration Plot

<img width="731" height="555" alt="image" src="https://github.com/user-attachments/assets/fc071c41-60c1-4583-b583-d6742d37cc51" />


### New Sample Data Prediction

<img width="331" height="44" alt="image" src="https://github.com/user-attachments/assets/dddd2c0a-ec0f-4bd7-a48a-5d485688ba36" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
