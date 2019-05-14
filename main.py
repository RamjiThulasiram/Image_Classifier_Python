import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def DataLoader(path):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir,transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform = valid_transforms)

    train_loaders = torch.utils.data.DataLoader(train_data,batch_size = 64,shuffle=True)
    test_loaders = torch.utils.data.DataLoader(test_data,batch_size = 32,shuffle = True)
    valid_loaders = torch.utils.data.DataLoader(valid_data,batch_size = 32,shuffle = True)

    return train_loaders,test_loaders,valid_loaders,train_data

def Neural_Network(arch,drop,hidden1,hidden2):
    '''
        Arguement: Pretained CNN, hyperparameters like dropout and no.of hidden layers
        Returns  : A Pretrained Network with user defined classifier

        This function takes a Pretrained CNN and combine it with a user defined class
     '''
    # TODO: Build and train your network

    architecture = {'resnet101':2048,
                  'densenet161':2208,
                  'vgg16':25088,
                  'alexnet':9216,
                  'densenet121':1024}

    if arch=="densenet121":
        model= models.densenet121(pretrained = True)
    elif arch=="alexnet":
        model= models.alexnet(pretrained = True)
    elif arch=="vgg16":
        model=models.vgg16(pretrained = True)
    elif arch=="densenet161":
        model=models.densenet161(pretrained = True)
    elif arch=="resnet101":
        model=models.resnet101(pretrained = True)
    else:
        print("Select One Architecture from the option")

    for param in model.parameters():
        param.requires_grad = False

    inputsize=architecture[arch]
    hidden = [hidden1,hidden2,90,102]

    classifier = nn.Sequential(OrderedDict([
                              ('input', nn.Linear(inputsize,hidden[0])),
                              ('relu1',nn.ReLU()),
                              ('fc1',nn.Linear(hidden[0],hidden[1])),
                              ('relu2',nn.ReLU()),
                              ('dropout',nn.Dropout(drop)),
                              ('fc2',nn.Linear(hidden[1],hidden[2])),
                              ('relu3',nn.ReLU()),
                              ('fc3',nn.Linear(hidden[2],hidden[3])),
                              ('softmax',nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    return model,criterion,optimizer


def do_deep_learning(model,device,trainloader,testloader,validloader, epochs, print_every, criterion, optimizer):

    '''
        Arguement   : model, Seperate Dataloaders for training,testing and validation,no. of epochs,
                      criterion and optimizer
        Return      : A Trained Neural_Network with Training Accuracy

        This function takes pretrained model with classifier and takes all hyperparameters,then trains
        them with desired device

    '''
    #---Start the Training---

    #Follwoing option lets you choose which device you want to work with....
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epoch = epochs
    print_every = print_every
    steps = 0

    for e in range(epochs):
        running_loss = 0
        for ii, (images, labels) in enumerate(trainloader):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                for ii, (test_images, test_labels) in enumerate(validloader):
                    test_images, test_labels = test_images.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.no_grad():
                        test_loss=0
                        accuracy=0
                        output=model.forward(test_images)
                        test_loss+=criterion(outputs,test_labels).item()
                        ps=torch.exp(outputs)
                        equality=(test_labels.data==ps.max(dim=1)[1])
                        accuracy+=equality.type(torch.FloatTensor).mean()

                print("Epoch: {}/{}... ".format(e+1, epoch),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                         "Test_Loss:{:3f}".format(test_loss/len(validloader)),
                         "Test Accuracy:{:3f}".format(accuracy/len(validloader)))

                running_loss = 0


    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))





def save_checkpoint(model,path,arch,drop,lr,epoch,hidden1,hidden2,train_data,optimizer):
    '''
        Arguement: Path to checkpoint save directory,Pretrained Network that the classifier uses,
        hyperparameters like dropout,Learning rate,No. of epochs,no.of hidden layers
        Returns  : Saves the progress in 'checkpoint.pth' file

        This function takes all hyperparameters and network as arguements and saves the progress
        in a seperate file 'checkpoint.pth'
    '''
    model.class_to_idx = train_data.class_to_idx

    checkpoint={'arch' :arch,
                'state_dict':model.state_dict(),
                'dropout':drop,
                'hidden layer - 1':hidden1,
                'hidden layer - 2':hidden2,
                'no_of_epochs':epoch,
                'optimizer':optimizer.state_dict(),
                'learning_rate':lr,
                'class_to_idx':model.class_to_idx}
    torch.save(checkpoint,path)

def load_checkpoint(file_path='checkpoint.pth'):
    '''
        Arguement: file_path
        Return   : Loads an already saved checkpoint

        Model of the Architecture with all the saved hyperparameters,weights and biases
    '''

    checkpoint = torch.load('checkpoint.pth')
    arch = checkpoint['arch']
    drop = checkpoint['dropout']
    hidden1 = checkpoint['hidden layer - 1']
    hidden2 = checkpoint['hidden layer - 2']
    model = models.vgg16()

    architecture = {'resnet101':2048,
                  'densenet161':2208,
                  'vgg16':25088,
                  'alexnet':9216,
                  'densenet121':1024}


    inputsize=architecture[arch]
    hidden = [hidden1,hidden2,90,102]

    classifier = nn.Sequential(OrderedDict([
                              ('input', nn.Linear(inputsize,hidden[0])),
                              ('relu1',nn.ReLU()),
                              ('fc1',nn.Linear(hidden[0],hidden[1])),
                              ('relu2',nn.ReLU()),
                              ('dropout',nn.Dropout(drop)),
                              ('fc2',nn.Linear(hidden[1],hidden[2])),
                              ('relu3',nn.ReLU()),
                              ('fc3',nn.Linear(hidden[2],hidden[3])),
                              ('softmax',nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    class_to_idx = checkpoint['class_to_idx']
    
    return model,class_to_idx

def process_image(image):
    '''
        Arguement :image
        Returns   :a np_array

        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image=Image.open(image)
    # TODO: Process a PIL image for use in a PyTorch model
    width = image.size[0]
    length = image.size[1]

    if length > width:
        image.thumbnail((256, length), Image.ANTIALIAS) #image = image.resize((256, length))
    else:
        image.thumbnail((width, 256), Image.ANTIALIAS)  #image = image.resize((width, 256))

                                             #Image Cropping size(as a box):
    image = image.crop((256/2 - 224/2,             #Left
                       256/2 - 224/2,              #Upper
                       256/2 + 224/2,              #Right
                       256/2 + 224/2))             #lower

    image = (np.array(image))/255

    mean = np.array([0.485, 0.456, 0.406])
    st_dev = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / st_dev
    image = image.transpose((2, 0, 1))


    return image

def predict(image_path, model, topk,device,class_to_idx,train_data):

    ''' Arguement:
            image_path,model,topk,device
        Return:
            The top (topk) Probablities of Classes of the Probable images

        Predict the class (or classes) of an image using a trained deep learning model.
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    model.class_to_idx=train_data.class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    model = model.to(device)


    np_array= process_image(image_path)
    img_tensor = torch.from_numpy(np_array)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.float()

    with torch.no_grad():
        outputs = model.forward(img_tensor.cuda())
   
    probabilities = torch.exp(outputs).data.topk(topk)
    probs = probabilities[0].tolist()
    classes = probabilities[1].tolist()
    class_idx = list()
    for i in classes[0]:
        class_idx.append(idx_to_class[i])
    return probs[0],class_idx
