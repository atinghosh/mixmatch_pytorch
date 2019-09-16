import torch
import os
import numpy as np
import imgaug.augmenters as iaa
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from network_mod import Wide_ResNet
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
lr = .01

def to_tensor(images, aug=None):
    tr = transforms.ToTensor()
    if aug is not None:
        images = aug(images=images)
    final_batch = [tr(im).unsqueeze(0) for im in images]
    return torch.cat(final_batch, 0).cuda()

def get_augmenter():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.ContrastNormalization((0.8, 1.3)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 2.0), per_channel=0.5),
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-10, 10),
            shear=(-5, 5)
        )
    ])

    def augment(images):
        tr = transforms.ToTensor()
        batch_numpy_tr_image = seq(images=images)
        final_batch = [tr(im).unsqueeze(0) for im in batch_numpy_tr_image]
        return torch.cat(final_batch, 0).cuda()
    return augment


def sharpen(x, T):
    temp = x ** (1 / T)
    return temp / temp.sum(dim=1, keepdim=True)


def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, alpha)
    beta = max([beta, 1 - beta])
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y


def mixmatch(x, y, u, model, augment_fn, T=0.5, K=2, alpha=0.75):
    model.eval()
    with torch.no_grad():
        xb = augment_fn(x)
        ub = [augment_fn(u) for _ in range(K)]
        output = [model(u) for u in ub]
        output = (output[0] + output[1])/2
        qb = sharpen(output, T)
        # qb = sharpen(sum(map(lambda i: model(i), ub)) / K, T)
    Ux = torch.cat(ub, dim=0)
    Uy = torch.cat([qb for _ in range(K)], dim=0)
    indices = np.arange(len(xb) + len(Ux))
    np.random.shuffle(indices)
    Wx = torch.cat([xb, Ux], dim=0)[indices]
    Wy = torch.cat([y, Uy], dim=0)[indices]

    X, p = mixup(xb, Wx[:len(xb)], y, Wy[:len(xb)], alpha)
    U, q = mixup(Ux, Wx[len(xb):], Uy, Wy[len(xb):], alpha)

    return X, U, p, q


class MixMatchLoss(torch.nn.Module):
    def __init__(self, lambda_u=75):
        super(MixMatchLoss, self).__init__()
        self.lambda_u = lambda_u
        self.logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
        self.mse = torch.nn.MSELoss().cuda()


    def forward(self, X, U, p, q, model):
        X_ = torch.cat([X,U], dim=0)
        preds = model(X_)
        sup_loss = torch.mean(torch.sum(- p * self.logsoftmax(preds[:len(p)]), 1))
        return sup_loss + self.lambda_u * self.mse(preds[len(p):], q)


def basic_generator(x, y=None, batch_size=32, shuffle=True):
    i = 0
    all_indices = np.arange(len(x))
    if shuffle:
        np.random.shuffle(all_indices)
    while (True):
        indices = all_indices[i:i + batch_size]
        if y is not None:
            yield x[indices], y[indices]
        else:
            yield x[indices]
        i = (i + batch_size) % len(x)


def mixmatch_wrapper(x, y, u, model, batch_size=32):
    augment_fn = get_augmenter()
    train_generator = basic_generator(x, y, batch_size)
    unlabeled_generator = basic_generator(u, batch_size=batch_size)
    while (True):
        xi, yi = next(train_generator)
        ui = next(unlabeled_generator)
        yield mixmatch(xi, yi, ui, model, augment_fn)


def to_torch(*args, device='cuda'):
    convert_fn = lambda x: torch.from_numpy(x).to(device)
    return list(map(convert_fn, args))

def test(model, test_gen, test_iters):
    acc = []
    model.eval()
    for i, (x, y) in enumerate(test_gen):
        x = to_tensor(x)
        pred = model(x).to('cpu').argmax(dim=1).numpy()
        acc.append(np.mean(pred == y))
        if i == test_iters:
            break
    print('Accuracy was : {}'.format(np.mean(acc)))


def report(loss_history):
    print('Average loss in last epoch was : {}'.format(np.mean(loss_history)))
    return []

def save(model, iter, train_iters):
    torch.save(model.state_dict(), 'model_{}.pth'.format(train_iters // iter))


def run(model, train_gen, test_gen, epochs, train_iters, test_iters):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MixMatchLoss()
    loss_history = []
    for i, (x, u, p, q) in enumerate(train_gen):
        if i % train_iters == 0:
            loss_history = report(loss_history)
            test(model, test_gen, test_iters)
            save(model, i, train_iters)
            if i // train_iters == epochs:
                return
        else:
            optim.zero_grad()
            loss = loss_fn(x, u, p, q, model)
            loss.backward()
            optim.step()
            loss_history.append(loss.to('cpu'))


if __name__ == "__main__":
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

    # Train set / Validation set split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_amount, random_state=1,
                                                              shuffle=True, stratify=y_train)

    # Train unsupervised / Train supervised split
    # Train set / Validation set split
    X_train, X_u_train, y_train, y_u_train = train_test_split(X_train, y_train, test_size=training_u_amount, random_state=1,
                                                              shuffle=True, stratify=y_train)

    X_remain, X_train, y_remain, y_train = train_test_split(X_train, y_train, test_size=training_amount, random_state=1,
                                                              shuffle=True, stratify=y_train)

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # model = Wide_ResNet(28, 10, 0.3, 10).cuda()
    model = Wide_ResNet(10).cuda()

    y_train = torch.tensor(y_train).cuda()
    y_train = torch.nn.functional.one_hot(y_train).float()

    train_generator = mixmatch_wrapper(X_train, y_train, X_u_train, model, 32)
    test_generator = basic_generator(X_remain, y_remain, 32)
    # run(model, train_generator, test_generator, 100, 20, )

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.00001)
    loss_fn = MixMatchLoss()
    final_loss = 0
    count = 0
    for i, (x, u, p, q) in enumerate(train_generator):
        model.train()
        loss = loss_fn(x, u, p, q, model)
        optim.zero_grad()
        loss.backward()
        optim.step()
        final_loss += loss.item()
        count += 1
        if i%100 == 0:
            print(f"loss is {final_loss/count}")
            test(model, test_generator, 50)
            model.train()



