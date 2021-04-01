# SimpleCNN
- This is the public repo of simpleCNN

## Quick exmple
```
python train_cifar10_classifier.py
python train_mnist_classifier.py
```
## Usage Example
### Build the Network
- Layer
  - You can use the code inside the **Layers** folder 
  - Eg. If you want to use 2D convolutional layer
  ```
  from Layers.Convolution2d import Conv2d
  ```
- Activation Function
  - You can use the code inside the **Activate** folder 
  - Eg. If you want to use ReLU
  ```
  from Activate.Activate import ReLU
  ```
### Loss Function
  - You can use the code in **Loss** filder
  - Eg. If you want to use HuberLoss
  ```
  from Loss.Loss import HuberLoss
  ```
### Load the Data
  - You can use the code in **Method** folder
  - Eg. Use Cifar10 dataset
  ```
  from Method.Data import Load_file
  train_x, train_y = Load_file('cifar10', 'train')
  test_x,test_y    = Load_file('cifar10', 'test')
  ```
