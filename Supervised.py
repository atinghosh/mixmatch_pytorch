import torch
import os
import numpy as np
import torch.nn as nn
import imgaug.augmenters as iaa
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from network_mod import Wide_ResNet
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from main import basic_generator, to_tensor, test
lr = .001

training_amount = 250
training_u_amount = 40000
validation_amount = 500

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=None)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=None)

X_train = np.array(trainset.data)
y_train = np.array(trainset.targets)

X_test = np.array(testset.data)
y_test = np.array(testset.targets)

# # Train set / Validation set split
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_amount, random_state=1,
#                                                           shuffle=True, stratify=y_train)
#
# # Train unsupervised / Train supervised split
# # Train set / Validation set split
# X_train, X_u_train, y_train, y_u_train = train_test_split(X_train, y_train, test_size=training_u_amount, random_state=1,
#                                                           shuffle=True, stratify=y_train)
#
# X_remain, X_train, y_remain, y_train = train_test_split(X_train, y_train, test_size=training_amount, random_state=1,
#                                                           shuffle=True, stratify=y_train)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# model = Wide_ResNet(28, 10, 0.3, 10).cuda()

y_train = torch.tensor(y_train).cuda()
# y_train = torch.nn.functional.one_hot(y_train).float()

# model = Wide_ResNet(28, 10, 0.3, 10).cuda()
model = Wide_ResNet(10).cuda()

optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.00001)

train_gen = basic_generator(X_train, y_train, 32)
test_gen = basic_generator(X_test, y_test, 32)

final_loss = 0
count = 0
loss_fn = nn.CrossEntropyLoss()
for i, (x, y) in enumerate(train_gen):
    x = to_tensor(x)
    y = torch.tensor(y).cuda().long()
    model.train()
    pred = model(x)
    optim.zero_grad()
    loss = loss_fn(pred, y)
    loss.backward()
    optim.step()
    final_loss += loss.item()
    count += 1
    if i % 100 == 0:
        print(f"loss is {final_loss / count}")
        test(model, test_gen, 50)
        model.train()
