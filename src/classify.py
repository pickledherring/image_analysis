import math
import torch
import numpy as np
import glob
import os
from re import search
from open_save import open_in_gray
from torch.utils.data import Dataset, random_split, DataLoader
from torch import nn

class KNN_Classifier():
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_pred):
        num_feats = len(self.X[0])
    
        preds = []
        for x in X_pred:
            # k_distances: [distance, index]
            k_distances = [[100000000, i] for i in range(self.k)]
            for i in range(len(self.X)):
                # find euclidean distance
                sq_diffs = [(x[j] - self.X[i][j])**2 for j in range(num_feats)]
                dist = math.sqrt(sum(sq_diffs))
                # compare to current neighbors
                for k in range(len(k_distances)):
                    if dist < k_distances[k][0]:
                        # shift up farther objects in the list
                        for k_r in range(len(k_distances) - 1, k, -1):
                            k_distances[k_r][0] = k_distances[k_r-1][0]
                            k_distances[k_r][1] = k_distances[k_r-1][1]
                        k_distances[k][0] = dist
                        k_distances[k][1] = i
            # evaluate final nearest neighbors            
            classes = {}
            for nn in k_distances:
                if str(self.y[nn[1]]) in classes.keys():
                    classes[str(self.y[nn[1]])] += 1
                else:
                    classes[str(self.y[nn[1]])] = 1

            # find first mode of the neighbors' classes
            max_class = str(self.y[k_distances[0][1]])
            max_value = classes[max_class]
            for key in classes.keys():
                if classes[key] > max_value:
                    max_class = key
                    max_value = classes[max_class]
                
            preds.append(max_class)

        return preds
    
class Cancer_Dataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = []
        # take all images from directory
        paths = glob.glob(f"{img_dir}/*")
        for _, path in enumerate(paths):
            name = search(r'smears.\D*', path).group(0)[7:]
            match name:
                case "cyl":
                    name_index = 0
                    # columnar epithelial?
                case "para":
                    name_index = 1
                    # parabasal squamous epithelial
                case "inter":
                    name_index = 2
                    # intermediate squamous epithelial
                case "super":
                    name_index = 3
                    # superficial squamous epithelial
                case "let":
                    name_index = 4
                    # mild nonkeratinizing dysplastic?
                case "mod":
                    name_index = 5
                    # moderate nonkeratinizing dysplastic
                case "svar":
                    name_index = 6
                    # severe nonkeratinizing dysplastic?
                case _:
                    name_index = 0
            
            item = [path.split("/")[-1], name_index]
            self.img_labels.append(item)
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        img = open_in_gray(img_path)
        img_tensor = torch.from_numpy(img)
        label = self.img_labels[idx][1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img_tensor, label
    
def split_data(data_loader, train_pct=.75, batch_size=64):
    train_size = int(train_pct * len(data_loader))
    test_size = len(data_loader) - train_size
    train_dataset, test_dataset = random_split(data_loader, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

class CNN(nn.Module):
    def __init__(self): 
        super(CNN, self).__init__()
        
        self.start = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),         
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
            nn.BatchNorm2d(num_features=12)           
        )
        
        self.end = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)   
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(140*190*12, 64),         
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(64, 7),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        x = self.start(x)
        x = self.end(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def run_cnn(train_loader, test_loader, n_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.NLLLoss() #TODO add weights
    learning_rate = 0.0025
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    batch_size = 64
    num_updates = n_epochs * int(np.ceil(len(train_loader) / batch_size))
    warmup_steps = 1000
    def warmup_linear(x):
        if x < warmup_steps:
            lr = x / warmup_steps
        else:
            lr = max((num_updates - x) / (num_updates - warmup_steps), 0.)
        return lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear)
    
    training_losses = []
    training_accs = []
    testing_losses = []
    testing_accs = []

    for i in range(n_epochs):
        training_loss = 0
        training_correct = 0
        testing_loss = 0
        testing_correct = 0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            #forward phase - predictions by the model
            outputs = model(inputs)
            #forward phase - risk/loss for the predictions
            loss = criterion(outputs, labels)
    
            # calculate gradients
            loss.backward()
            
            # take the gradient step
            optimizer.step()
            scheduler.step()

            training_loss += loss.item()
            pred = outputs.data.max(dim=1, keepdim=True)[1]
            training_correct += pred.eq(labels.data.view_as(pred)).sum().item()

        with (torch.no_grad()):
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                testing_loss += loss.item()

                pred = outputs.data.max(dim=1, keepdim=True)[1]
                testing_correct += pred.eq(labels.data.view_as(pred)).sum().item()

        avg_train_loss = training_loss / len(train_loader)
        train_acc = training_correct / len(train_loader.dataset)
        training_losses.append(avg_train_loss)
        training_accs.append(train_acc)

        avg_test_loss = testing_loss / len(test_loader)
        test_acc = testing_correct / len(test_loader.dataset)
        testing_losses.append(avg_test_loss)
        testing_accs.append(test_acc)

        print(f"{i}: training: loss = {avg_train_loss}")
        print(f"\t\tacc = {train_acc}")
        print(f"{i}testing: loss = {avg_test_loss}")
        print(f"\t\tacc = {test_acc}")
        #TODO better outputs

    return (training_losses, training_accs, testing_losses, testing_accs)