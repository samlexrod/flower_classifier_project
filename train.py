#!/usr/bin/env python
# coding: utf-8

# # all imports for training

# In[1]:


# D:\GitHub\Flower_Classifier_Project\flower_data
# D:\GitHub\Flower_Classifier_Project


# In[1]:


from torchvision import datasets, transforms
from torch import nn, optim
from collections import OrderedDict
import torch
import torchvision.models as models
from datetime import datetime as dt
import pickle
import argparse
import os


# In[3]:


parser = argparse.ArgumentParser()

parser.add_argument('data_dir', metavar='data_dir', type=str, nargs='?',
                    help='This is the root directory of the images.')

parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='this is the path where the checkpoint.pth file will be located.')

parser.add_argument('--arch', action='store', type=str,
                    dest='arch',
                    help='Provide the architecture of the model vgg19_bn or densenet121.')

parser.add_argument('--learning_rate', action='store', type=float,
                    dest='learning_rate',
                    help='Provide the learning rate of the model in decimal points.')

parser.add_argument('--hidden_units', action='store', type=str,
                    dest='hidden_units',
                    help='Provide the hidden units of the model 500,300,etc. --no space')

parser.add_argument('--epochs', action='store', type=int,
                    dest='epochs',
                    help='Provide the epochs of the training as an int.')

parser.add_argument('--gpu', action="store_true", default=False,
                   help='Set the model to predicut using the GPU')

'''parser.add_argument('--device', action='store', type=str,
                    dest='device',
                    help='Provide the device cuda or cpu in which the model will train.')'''

results = parser.parse_args()

if results.gpu:
    results.device = 'cuda'
else:
    results.device = 'cpu'

if results.hidden_units:
    results.hidden_units = [int(x) for x in results.hidden_units.split(',')]

print('\nCommand line selections:')
print('Data Directory = {!r}'.format(results.data_dir))
print('Checkpoint save_dir = {!r}'.format(results.save_dir))
print('Architecture = {!r}'.format(results.arch))
print('Learning Rate = {!r}'.format(results.learning_rate))
print('Hidden Units = {!r}'.format(results.hidden_units))
print('Epochs = {!r}'.format(results.epochs))
print('Device = {!r}'.format(results.device))
print('-')

'''print('Command line warnings:')
if results.image_path == None: print('No image parsed. You will have the chance to select an image as an input.')
if results.checkpoint_name == None: print('No checkpoint name given. The default cehckpoint.pth will be used. ')
if results.cat_to_name == None: print('No --cat parsed. You will have the chance to input categories later.')
if results.topk == None: print('No --topk parsed. You will have the chance to input the top k later.')
if results.device == None: print('No --device parsed. You will have the chance to input the device later.')
print('-')'''

# command line checks
print('Command line warnings:')
if results.device not in ('cuda', 'cpu', None):
    results.device = None
    print('Wrong device input. Please use the command input to select desired device.')
if results.arch not in ('vgg19_bn', 'densenet121', None):
    results.arch = None
    print('Wrong model architecture. Please use the command input to select desired architecture.')
if type(results.hidden_units) != list and results.hidden_units != None:
    results.hidden_units = None
    print('Wrong hidden units format. Please use the command input the hidden units.')


# # preping data folders

# In[2]:


#class results: pass


# In[21]:


results.data_dir = r'D:\GitHub\Flower_Classifier_Project\flower_data'
# setting data directory
if results.data_dir == None:
    data_dir = input('What is the root path of the data? - default is flower_data ->')
    if data_dir == '' or data_dir.lower() == 'default': data_dir = 'flower_data'
else:
    data_dir = results.data_dir
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')
print('-\nModel data folders:')
print(
'    Using root path: \t{}\n\
    Using train path: \t{}\n\
    Using valid path: \t{}\n\
    Using test path: \t{}'.format(data_dir, train_dir, valid_dir, test_dir))


# # data transformer

# In[22]:


means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

# TODO: Define your transforms for the training, validation, and testing sets
train_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=means, std=stds)])

test_data_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=means, std=stds)])

valid_data_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=means, std=stds)])


# TODO: Load the datasets with ImageFolder
train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
test_image_datasets = datasets.ImageFolder(train_dir, transform=test_data_transforms)
valid_image_datasets = datasets.ImageFolder(test_dir, transform=valid_data_transforms)
class_to_tensoridx_dict = train_image_datasets.class_to_idx

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=40, shuffle=True)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=40, shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=16, shuffle=True)
                                           
dataloaders = {'train': train_dataloaders, 'test': test_dataloaders, 'valid': valid_dataloaders}
                                           
print('Great! Your test images have been transformed.')
print('-')


# # Questionnaire

# In[23]:


#results.device = 'cuda'
def device_validation():
    
    if results.device == None:
        if torch.cuda.is_available():    
            device = input('In what device do you want to run this model, cuda or cpu? ->')
            if device.lower() in ['cpu', 'cuda']:
                print('Thanks! You selected to run the model using the {}.'.format(device))
                return device
            else:
                print('Warning! Wrong input. Choose cuda or cpu.')
                return device_validation()
        else:
            device = 'cpu'
            print('Sorry! But your device does not support GPU. Note that the training will run faster with a GPU.             Please, consider changing to a device with a GPU.')
    else:
        return results.device
device = device_validation()
print('-')


# In[24]:


#results.epochs = 1
def epoch_validation():
    
    if results.epochs == None:
        try:
            epochs = int(input('How many epoch do you want to run?             \nenter int value ->'))
            print('Thanks! You have selected to run {} epochs.'.format(epochs))
            return epochs
        except:
            print('Warning! Enter only integer values')   
            return epoch_validation()
    else:
        return results.epochs
    
epochs = epoch_validation()
print('-')


# In[25]:


def print_sequence_validation():
    try:
        print_sequence = int(input('How many steps before printing the epoch and loss?         \nenter int value ->'))
        print('Thanks! You have selected to print the epochs and loss on every {} steps.'.format(print_sequence))
        return print_sequence
    except:
        print('Warning! Enter only integer values')
        return print_sequence_validation()
    
print_sequence = print_sequence_validation()  
print('-')


# In[26]:


#results.learning_rate = 0.001
def learning_rate_validation():
    
    if results.learning_rate == None:    
        try:
            learning_rate = float(input('What is the learning rate desired?             \nenter float value ->'))
            print('Thanks! You have selected use a learning rate of {:f} steps.'.format(learning_rate))
            return learning_rate
        except:
            print('Warning! Enter only float numbers')
            return learning_rate_validation()
    else:
        return results.learning_rate
    
learning_rate = learning_rate_validation()
print('-')


# In[3]:


#results.arch = 'densenet121'
def choose_model():
    
    def set_model_input_name():
        if model_selected in ['1', 'densenet121']: 
            model = models.densenet121(pretrained=True)
            input_features = model.classifier.in_features   
            model_name = 'densenet121'
            print('Great! You have selected the densenet121 architecture.')
            print('The model input layer is', input_features)
            return model, input_features, model_name
        elif model_selected in ['2', 'vgg19_bn']: 
            model = models.vgg19_bn(pretrained=True)
            input_features = model.classifier[0].in_features
            model_name = 'vgg19_bn'
            print('Great! You have selected the vgg19_bn architecture.')
            print('The model input layer is', input_features)
            return model, input_features, model_name
        else:
            print('Wanring! Please select a valid architecture: 1 for densenet121 or 2 for vgg19_bn.')
            return choose_model()
    
    if results.arch == None:           
        print('Great! We have two great models to run: 1 for densenet121 or 2 for vgg19_bn')
        model_selected = str(input('Which model will you choose, 1 or 2? ->'))
        return set_model_input_name()
    else:
        model_selected = results.arch
        return set_model_input_name()

model, input_features, model_name = choose_model()   
print('-')


# In[4]:


def hidden_layer_generator(hidden_layer_qty, model_input_features, hidden_layer_inputs, output_dim):
    """
    Parameters
    ----------
    hidden_layer_qty: the number of hidden layers -required
    model_input_features: the model imput features comming from the selected model -required
    hidden_layer_inputs: the hidden layer imputs in a list [input_int, input_int2] -required
    output_dim: the dimensions of the LogSoftmax output -default 1
    
    returns the model classifier
        assign it by using model.classifier = hidden_layer_generator(parameters_here)
    """
    if hidden_layer_qty == len(hidden_layer_inputs):
        
        # initiating layer list
        hidden_layers = [('fc1', nn.Linear(input_features , hidden_layer_inputs[0])), 
                         ('relu', nn.ReLU()),
                         ('dropout', nn.Dropout(p=0.5))]
        
        # generating hidden layers and output  
        for number in range(hidden_layer_qty):
            fc = number + 2
            layer_name = ('fc%s' % fc)
            try:
                layer = nn.Linear(hidden_layer_inputs[number], hidden_layer_inputs[number+1])
            except:
                layer = nn.Linear(hidden_layer_inputs[number], 102)
            hidden_layers.append((layer_name, layer))
            if number < range(hidden_layer_qty)[-1]:
                hidden_layers.append(('relu', nn.ReLU()))
                hidden_layers.append(('dropout', nn.Dropout(p=0.5)))
            else:
                hidden_layers.append(('output', nn.LogSoftmax(dim=output_dim)))

    else:
        print('The lenght of the list of hidden layers does not equal the quantity of layers.')
        
    return hidden_layers


# In[16]:


#results.hidden_units = None
def layer_validation():
    
    def layer_treshold_violation(layer_argument):
        if layer_argument[0] >= input_features*.95:
            print('Hidden layers exceeds input layer of %s. Please use the command input to enter the hidden layers.' % input_features)
        return layer_argument[0] >= input_features*.95
    
    def layer_arguments():
        try:
            hidden_layer_inputs = input('Enter a list of hidden inputs separated by 1 space             \n(E.g. 500 300 120) ->').split(' ')
            hidden_layer_inputs = [int(num) for num in hidden_layer_inputs]
            
            # hidden layer validation
            if layer_treshold_violation(hidden_layer_inputs):
                return layer_arguments()
            else:
                return len(hidden_layer_inputs), hidden_layer_inputs, 1 
        except:
            print('Warning! Unexpedted input character, please enter integers')
            print("Let's try that again: \n")
            #return layer_arguments()

    
    if results.hidden_units == None:
        return layer_arguments()
    else:
        if layer_treshold_violation(results.hidden_units):
            results.hidden_units = None
            return layer_arguments()
        else:
            return len(results.hidden_units), results.hidden_units, 1
    
# calling funciton
hidden_layer_qty, hidden_layer_inputs, ouput_dim = layer_validation()
    
hidden_layers = hidden_layer_generator(hidden_layer_qty, 
                                                input_features, 
                                                hidden_layer_inputs, 
                                                ouput_dim)
ordered_dict = OrderedDict(hidden_layers)

classifier = nn.Sequential(ordered_dict)

print('-')
print('\nGreat! Your classifier is ready:')
print(classifier)
print('-')


# # Helper Clock

# In[34]:


class TickTock:
    """
    Automatic timer. Assign it to a variable and call the stop_clock method.
    Returns a string format '0:00:00.000000'
    """
    def __init__(self):        
        self.start_time = dt.now()
    def stop_clock(self):
        td = dt.now() - self.start_time
        return ':'.join(str(td).split(':'))


# # Training Model

# In[35]:


def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


# In[36]:


print('The model is now training. :)')
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = classifier
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

timer = TickTock()

for e in range(epochs):    
    running_loss = 0
    for steps, (images, labels) in enumerate(dataloaders['train']):
        model.train()
        
        # move images and labels to device selected
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_sequence == 0:
            model.eval()
            
            with torch.no_grad():
                test_loss, accuracy = validation(model, dataloaders['test'], criterion)
            
            
            print("Epoch: {}/{} -|- ".format(e+1, epochs),
                  "Train Loss: {:.4f} -|- ".format(running_loss/print_sequence),
                  "Test Loss: {:.3f} -|- ".format(test_loss/len(dataloaders['test'])),
                  "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))
            running_loss = 0   

time_delta = timer.stop_clock()
print('Training time:', time_delta)
print('-')


# # Saving Checkpoint

# In[49]:


#results.save_dir = r'D:\GitHub'


# In[50]:


if results.save_dir:
    checkpoint_path = results.save_dir + '\\checkpoint.pth'
else:
    checkpoint_path = 'checkpoint.pth'


# In[52]:


model.class_to_idx = class_to_tensoridx_dict
checkpoint = {'class_to_idx': model.class_to_idx, 
              'classifier': model.classifier, 
              'model': model_name, 
              'state_dict': model.state_dict()} 

torch.save(checkpoint, checkpoint_path)
print('Checkpoint saved with {}'.format(', '.join(list(checkpoint.keys())[:-1])+', and '+list(checkpoint.keys())[-1] ))
print('-')


# # Testing Accuracy of Model with Testing Set

# In[15]:


print('We are now checking the accuracy of the model. :)')
correct = 0
total = 0
model.to(device)
with torch.no_grad():
    for (images, labels) in test_dataloaders:
        # move images and labels to device selected
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Training is done.')
print('Here is the accuracy of the network on test images: %d %%' %
     (100 * correct / total)) 


# In[ ]:




