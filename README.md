# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.
<img width="757" height="707" alt="image" src="https://github.com/user-attachments/assets/f0540ef4-089f-42c2-9e0d-282f3223ed30" />


## DESIGN STEPS
 ### STEP 1: 

Import the required Python libraries and load the customer segmentation dataset.

### STEP 2: 
Preprocess the dataset by removing unnecessary columns, handling missing values, encoding categorical attributes, and scaling the numerical features.


### STEP 3: 
Split the dataset into training and testing sets and convert them into PyTorch tensors.


### STEP 4: 
Design a feedforward neural network with three hidden layers using ReLU activation and an output layer containing four neurons.

### STEP 5: 

Train the neural network using the Cross-Entropy loss function and Adam optimizer for the specified number of epochs.

### STEP 6: 

Evaluate the trained model using the test dataset and generate the accuracy, confusion matrix, and classification report.




## PROGRAM

### Name: S.RISITHA

### Register Number: 212225240119

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/customers (1).csv')
data.head()

data.columns

# Drop ID column as it's not useful for classification
data = data.drop(columns=["ID"])

# Handle missing values
data.fillna({"Work_Experience": 0, "Family_Size": data["Family_Size"].median()}, inplace=True)

# Encode categorical variables
categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Encode target variable
label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])  # A, B, C, D -> 0, 1, 2, 3

# Split features and target
X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x):
      #Include your code here
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
  for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Initialize model
model = PeopleClassifier(X_train.shape[1])
criterion =  nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name: S.RISITHA")
print("Register No: 212225240119")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')
```




### Dataset Information
Include screenshot of the dataset.

<img width="1541" height="557" alt="image" src="https://github.com/user-attachments/assets/750f2146-b721-471f-9cb8-9f67272175c1" />


### OUTPUT

## Confusion Matrix

Include confusion matrix here

<img width="802" height="672" alt="image" src="https://github.com/user-attachments/assets/c1e302de-73d3-44b1-8cdd-101705acfcb0" />


## Classification Report
Include classification report here

<img width="857" height="513" alt="image" src="https://github.com/user-attachments/assets/7727b7f2-bf4c-42cf-97af-a267a817f1d5" />


### New Sample Data Prediction
Include your sample input and output here

<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/97182209-f10f-4675-9c80-91560074c4b9" />


## RESULT

The neural network classification model was successfully developed and trained using the customer segmentation dataset. The model effectively classified customers into the four segments (A, B, C, and D). The performance was evaluated using accuracy, confusion matrix, and classification report, demonstrating that the model is suitable for customer segmentation and prediction.

