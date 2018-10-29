#!/usr/bin/env python
# coding: utf-8

# # all imports for training

# In[23]:


from torchvision import datasets, transforms
from torch import nn, optim
from collections import OrderedDict
import torch
import torchvision.models as models
from datetime import datetime as dt
import pickle


# # preping data folders

# In[2]:


data_dir = input('What is the root path of the data? - default is flower_data ->')
if data_dir == '' or data_dir.lower() == 'default': data_dir = 'flower_data'
train_dir = data_dir +'/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
print(
'    Using root path: \t{}\n\
    Using train path: \t{}\n\
    Using valid path: \t{}\n\
    Using test path: \t{}'.format(data_dir, train_dir, valid_dir, test_dir))


# # data transformer

# In[3]:


means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=means, std=stds)])

test_data_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=means, std=stds)])

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
test_image_datasets = datasets.ImageFolder(train_dir, transform=test_data_transforms)
class_to_tensoridx_dict = image_datasets.class_to_idx

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=40, shuffle=True)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=40, shuffle=True)
print('Great! Your test images have been transformed.')
print('-')


# # Questionnaire

# In[4]:


def device_validation():
    device = input('In what device do you want to run this model, cuda or cpu? ->')
    if device.lower() in ['cpu', 'cuda']:
        print('Thanks! You selected to run the model using the {}.'.format(device))
        return device
    else:
        print('Warning! Wrong input. Choose cuda or cpu.')
        return device_validation()
device = device_validation()
print('-')


# In[5]:


def epoch_validation():
    try:
        epochs = int(input('How many epoch do you want to run?         \nenter int value ->'))
        print('Thanks! You have selected to run {} epochs.'.format(epochs))
        return epochs
    except:
        print('Warning! Enter only integer values')   
        return epoch_validation()
    
epochs = epoch_validation()
print('-')


# In[6]:


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


# In[7]:


def learning_rate_validation():
    try:
        learning_rate = float(input('What is the learning rate desired?         \nenter float value ->'))
        print('Thanks! You have selected use a learning rate of {:f} steps.'.format(learning_rate))
        return learning_rate
    except:
        print('Warning! Enter only float numbers')
        return learning_rate_validation()
learning_rate = learning_rate_validation()
print('-')


# In[8]:


def choose_model():
    print('Great! We have two great models to run: 1 for densenet121 or 2 for vgg19_bn')
    model_selected = str(input('Which model will you choose, 1 or 2? ->'))
    if model_selected in ['1', 'densenet121']: 
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features        
        print('Great! You have selected the densenet121 architecture.')
        print('The model input layer is', input_features)
        return model, input_features
    elif model_selected in ['2', 'vgg19_bn']: 
        model = models.vgg19_bn(pretrained=True)
        input_features = model.classifier[0].in_features
        print('Great! You have selected the vgg19_bn architecture.')
        print('The model input layer is', input_features)
        return model, input_features
    else:
        print('Wanring! Please select a valid architecture: 1 for densenet121 or 2 for vgg19_bn.')
        return choose_model()

model, input_features = choose_model()   
print('-')


# In[10]:


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
                layer = nn.Linear(hidden_layer_inputs[number], 121)
            hidden_layers.append((layer_name, layer))
            if number < range(hidden_layer_qty)[-1]:
                hidden_layers.append(('relu', nn.ReLU()))
                hidden_layers.append(('dropout', nn.Dropout(p=0.5)))
            else:
                hidden_layers.append(('output', nn.LogSoftmax(dim=output_dim)))

    else:
        print('The lenght of the list of hidden layers does not equal the quantity of layers.')
        
    return hidden_layers


# In[12]:


def layer_validation():
    try:
        hidden_layer_qty = int(input('How many layers in the model?         \nenter int number ->'))
        hidden_layer_inputs = input('Enter a list of hidden inputs separated by 1 space         \n(E.g. 500 300 120) ->').split(' ')
        hidden_layer_inputs = [int(num) for num in hidden_layer_inputs]
        
        # quantity of layers vs hidden layer list validation
        if hidden_layer_qty != len(hidden_layer_inputs):            
            print('Warning! The lenght of the list of hidden layers does not equal the quantity of layers.')
            print("Let's try that again: \n")
            return layer_validation()
        
        ouput_dim = int(input('How many dimensions in the LogSoftmax output?         \nenter int number ->'))
        return hidden_layer_qty, hidden_layer_inputs, ouput_dim 
    except:
        print('Warning! Unexpedted input character, please enter integers')
        print("Let's try that again: \n")
        return layer_validation()

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

# In[9]:


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


# # Setting Model and Saving Neural Network Sequential Arguments

# In[16]:


model_settings = {'model': model, 'sequential_arg': ordered_dict}
with open('model_settings.pickle', 'wb') as handle:
    pickle.dump(model_settings, handle)


# # Training Model

# In[13]:


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
    for steps, (images, labels) in enumerate(dataloaders):

        # move images and labels to device selected
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_sequence == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                 "Loss: {:.4f}".format(running_loss/print_sequence))
            running_loss = 0

time_delta = timer.stop_clock()
print('Time to train: ', time_delta)
print('-')


# # Saving Mappings and Checkpoint

# In[14]:


with open('class_to_tensoridx_dict.pickle', 'wb') as handle:
    pickle.dump(class_to_tensoridx_dict, handle)

torch.save(model.state_dict(), 'checkpoint.pth')
print('We have saved a pickle file of classes to tensor index mappings and a checkpoint of the model.')
print('We have dump the model_settings pickle file to use in the prediciton.')
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

