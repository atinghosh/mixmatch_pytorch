# # Implementation of the paper   Generating Images with Perceptual Similarity Metrics based on Deep Networks
# # https://arxiv.org/pdf/1602.02644.pdf.
# # All the architectures are exactly same as mentioned in the above paper.
# # CURRENT best result for reconstruction but sampling is not working, output is
# # /data02/data02/Atin/SSL-autoencoder/weights/autoencoder_STL10_v2.pkl is the trained model.
#
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.utils import save_image
# from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
# import pickle
# import zipfile
# import datetime
# from torchvision.models import vgg19, alexnet
# import numpy as np
# import math
# num_epochs = 20
# real_label = 1
# fake_label = 0
#
#
# def calculate_size(input_size, kernel_size, stride, padding=0, transposed=False):
#
#     if transposed:
#         output_size = stride * (input_size - 1) + kernel_size - 2 * padding
#     else:
#         output_size = math.floor((input_size + 2 * padding - kernel_size)/stride) + 1
#     return output_size
#
#
# class Discriminator(nn.Module):
#
#     def __init__(self):
#
#         super(Discriminator, self).__init__()
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(3, 32, 7, 2, 0),  # 29
#             nn.LeakyReLU(.3),
#             nn.Conv2d(32, 64, 5, 1, 0),  # 25
#             nn.LeakyReLU(.3),
#             nn.Conv2d(64, 128, 3, 2, 0),  # 12
#             nn.LeakyReLU(.3),
#             nn.Conv2d(128, 256, 3, 1, 0),  # 10
#             nn.LeakyReLU(.3),
#             nn.Conv2d(256, 256, 3, 2, 0),  # 4
#             nn.LeakyReLU(.3),
#             nn.Conv2d(256, 256, 4, 4, 0),  # 256 x 1 x 1
#             nn.LeakyReLU(.3),
#         )
#         self.dropout = nn.Dropout(.5)
#         self.fc1 = nn.Linear(256, 512)
#         self.fc2 = nn.Linear(512, 2)
#
#     def forward(self, x):
#
#         x = self.conv_layer(x).view(-1, 256)
#         x = self.dropout(x)
#         x = F.leaky_relu(self.fc1(x), negative_slope=.3)
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x
#
#
# class Autoencoder(nn.Module):
#
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, 5, 2, 2),  # 32
#             nn.LeakyReLU(.3),
#             nn.Conv2d(32, 32, 3, 1, 1),
#             nn.LeakyReLU(.3),
#             nn.Conv2d(32, 64, 5, 2, 2),  # 16
#             nn.LeakyReLU(.3),
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.LeakyReLU(.3),
#             nn.Conv2d(64, 128, 3, 2, 1),  # 8
#             nn.LeakyReLU(.3),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.LeakyReLU(.3),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.LeakyReLU(.3),
#             nn.Conv2d(64, 8, 3, 1, 1),
#             # nn.LeakyReLU(.3),
#         )
#         self.decoder = nn.Sequential(
#             nn.LeakyReLU(.3),
#             nn.Conv2d(8, 64, 3, 1, 1),
#             nn.LeakyReLU(.3),
#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.LeakyReLU(.3),
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16
#             nn.LeakyReLU(.3),
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.LeakyReLU(.3),
#             nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32
#             nn.LeakyReLU(.3),
#             nn.Conv2d(32, 32, 3, 1, 1),
#             nn.LeakyReLU(.3),
#             nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 64
#             nn.LeakyReLU(.3),
#             nn.Conv2d(16, 3, 3, 1, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded
#
#
# def main():
#     transform = transforms.Compose(
#         [transforms.RandomCrop(64),
#          transforms.ToTensor(), ])
#     trainset = STL10(root='/data02/Atin/STL10/', split="train+unlabeled",
#                                           download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
#                                               shuffle=False, num_workers=10)
#
#     autoencoder = Autoencoder().cuda()
#     alexnet_model = alexnet(pretrained=True).cuda()
#     netD = Discriminator().cuda()
#
#     ce_criterion = nn.CrossEntropyLoss().cuda()
#     bce_criterion = nn.BCELoss().cuda()
#     mse_criterion = nn.MSELoss().cuda()
#
#     optimizer = optim.Adam(autoencoder.parameters(), lr=.0002)
#     optimizerD = optim.Adam(netD.parameters(), lr=.0002)
#
#     for epoch in range(20):
#         fina_running_loss = 0.0
#         count = 0
#         for i, (inputs, _) in enumerate(trainloader, 0):
#             inputs = inputs.cuda()
#             b_size = inputs.shape[0]
#
#             # ============ Forward ============
#             encoded, decoded = autoencoder(inputs)
#
#             # pixel loss
#             pixel_loss = bce_criterion(decoded, inputs)
#
#             # Perceptual loss
#             new_model = nn.Sequential(*list(alexnet_model.features.children())[:10]).cuda()
#             new_model.eval()
#             input_features = new_model(inputs)
#             output_features = new_model(decoded)
#             perceptual_loss = mse_criterion(input_features, output_features)
#
#             # Adverserial loss
#
#             # Discriminator loss
#             netD.zero_grad()
#             label1 = torch.full((b_size,), real_label).cuda().long()
#             label2 = torch.full((b_size,), fake_label).cuda().long()
#             label = torch.cat((label1, label2), 0).long()
#             real_plus_fake_dataset = torch.cat((inputs, decoded.detach()), 0)
#             discriminator_output = netD(real_plus_fake_dataset)
#
#             discriminator_output_post_softmax = F.softmax(discriminator_output)
#             D_x = discriminator_output_post_softmax[:b_size, 1]
#             D_x = D_x.mean().item()
#             D_G_z1 = discriminator_output_post_softmax[b_size:, 1]
#             D_G_z1 = D_G_z1.mean().item()
#
#             discriminator_loss = ce_criterion(discriminator_output, label)
#             discriminator_loss.backward()
#             optimizerD.step()
#
#             # Generative/Decoder loss
#             discriminator_output = netD(decoded)
#
#             discriminator_output_post_softmax = F.softmax(discriminator_output)
#             D_G_z2 = discriminator_output_post_softmax[:, 1]
#             D_G_z2 = D_G_z2.mean().item()
#
#             adv_loss = ce_criterion(discriminator_output, label1)
#
#             # Final combined loss
#             optimizer.zero_grad()
#             final_loss = pixel_loss + perceptual_loss + adv_loss
#             final_loss.backward()
#             optimizer.step()
#
#             fina_running_loss += final_loss.item()
#             count += 1
#             if i % 50 == 0:
#                 print(f"COMBINED loss is {fina_running_loss/count}")
#                 print(f"adv loss is {adv_loss.item()}")
#                 print(f"recon loss is {pixel_loss.item()}")
#                 print(f"perceptual loss is {perceptual_loss.item()}")
#                 print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
#                       % (epoch, num_epochs, i, len(trainloader),
#                          discriminator_loss.item(), adv_loss.item(), D_x, D_G_z1, D_G_z2))
#
#     print('Finished Training')
#     print('Saving Model...')
#     if not os.path.exists('./weights'):
#         os.mkdir('./weights')
#     torch.save(autoencoder.state_dict(), "./weights/autoencoder_STL10_v2.pkl")
#
#
# if __name__ == '__main__':
#     main()
#

# import numpy as np
# def basic_generator(x, y=None, batch_size=32, shuffle=True):
#     i = 0
#     all_indices = np.arange(len(x))
#     if shuffle:
#         np.random.shuffle(all_indices)
#     while (True):
#         indices = all_indices[i:i + batch_size]
#         if y is not None:
#             yield x[indices], y[indices]
#         yield x[indices]
#         i = (i + batch_size) % len(x)
#
# x = np.arange(100)
# y = np.arange(100)
# gen = basic_generator(x, y=y, batch_size=20, shuffle=True)
# for i, x in enumerate(gen):
#     pass


# import torch
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F
# from torch.autograd import Variable
#
# import sys
# import numpy as np
#
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
#
# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.xavier_uniform(m.weight, gain=np.sqrt(2))
#         init.constant(m.bias, 0)
#     elif classname.find('BatchNorm') != -1:
#         init.constant(m.weight, 1)
#         init.constant(m.bias, 0)
#
# class wide_basic(nn.Module):
#     def __init__(self, in_planes, planes, dropout_rate, stride=1):
#         super(wide_basic, self).__init__()
#         # self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
#         self.dropout = nn.Dropout(p=dropout_rate)
#         # self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
#             )
#
#     def forward(self, x):
#         out = self.dropout(self.conv1(F.relu(x)))
#         out = self.conv2(F.relu(out))
#         out += self.shortcut(x)
#
#         return out
#
# class Wide_ResNet(nn.Module):
#     def __init__(self, depth, widen_factor, dropout_rate, num_classes):
#         super(Wide_ResNet, self).__init__()
#         self.in_planes = 16
#
#         assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
#         n = (depth-4)/6
#         k = widen_factor
#
#         print('| Wide-Resnet %dx%d' %(depth, k))
#         nStages = [16, 16*k, 32*k, 64*k]
#
#         self.conv1 = conv3x3(3,nStages[0])
#         self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
#         self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
#         self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
#         # self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
#         self.linear = nn.Linear(nStages[3], num_classes)
#
#     def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
#         strides = [stride] + [1]*(int(num_blocks)-1)
#         layers = []
#
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, dropout_rate, stride))
#             self.in_planes = planes
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.relu(out)
#         out = F.avg_pool2d(out, 8)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class Wide_ResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(Wide_ResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)




if __name__ == '__main__':
    net=Wide_ResNet(10)
    y = net(torch.randn(1,3,32,32))

    print(y.size())