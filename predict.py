#!/usr/bin/env python
# coding: utf-8

# In[23]:


import torch
import os
from torchvision import transforms
from matplotlib import pyplot as plt
from torch import nn, optim
import numpy as np
import pickle
import json
import pandas as pd
from IPython.display import display
import numpy as np
import string
from PIL import Image
import argparse
import torchvision
from matplotlib import cm


# In[51]:


parser = argparse.ArgumentParser()

parser.add_argument('image_path', metavar='image_path', type=str, nargs='?',
                    help='Store the name of the json file with category names of the flowers.')

parser.add_argument('checkpoint_name', metavar='checkpoint_name', type=str, nargs='?',
                    help='Store the name of the json file with category names of the flowers.')

parser.add_argument('--cat', action='store',
                    dest='cat_to_name',
                    help='Store the name of the json file with category names of the flowers.')

parser.add_argument('--topk', action='store', type=int,
                    dest='topk',
                    help='Store the top k predictions wanted in the output.')

parser.add_argument('--device', action='store',
                    dest='device',
                    help='Store the device in which the prediction will run.')

results = parser.parse_args()

if results.checkpoint_name:
    results.checkpoint_name = str(results.checkpoint_name)+'.pth'

print('\nCommand line selections:')
print('image_path = {!r}'.format(results.image_path))
print('checkpoint_name = {!r}'.format(results.checkpoint_name))
print('cat_to_name = {!r}'.format(results.cat_to_name))
print('topk = {!r}'.format(results.topk))
print('device = {!r}'.format(results.device))
print('-')

print('Command line warnings:')
if results.image_path == None: print('No image parsed. You will have the chance to select an image as an input.')
if results.checkpoint_name == None: print('No checkpoint name given. The default cehckpoint.pth will be used. ')
if results.cat_to_name == None: print('No --cat parsed. You will have the chance to input categories later.')
if results.topk == None: print('No --topk parsed. You will have the chance to input the top k later.')
if results.device == None: print('No --device parsed. You will have the chance to input the device later.')
print('-')

# command line checks
if results.device not in ('cuda', 'cpu'):
    results.device = None
    print('Wrong device input. Please use the command input to select desired device.')
if results.topk:
    if results.topk >= 34 or results.topk <=2:
        results.topk = None
        print('Wrong topk input. Please use the command input to select desired topk from 2 to 33.')


# In[25]:


'''# debug
class results:
    pass
results.cat_to_name = None
results.topk = 5
results.device = 'cuda'
# debug'''


# # Helper Functions

# In[26]:


def pretty_flower_frame(cat_to_name):
    """
    A clean way to see the flower mappings to be able to validate the model easier
    
    Parameters
    ----------
    cat_to_name: a dictionary mapping the integer encoded categories to the actual names of the flowers
    
    display dataframe
    returns None
    """
    is_tensor = [True for value in cat_to_name.values() if value == 0]
    is_tensor = len(is_tensor) > 0
    
    # setting key names
    if is_tensor:
        key_name = 'folder_number'
        value_name = 'tensor_id'
    else:
        key_name = 'index'
        value_name = 'flower_name'
    
    # creating dict
    flower_dict = {key_name: [], value_name: []}
    for key, value in cat_to_name.items():
        flower_dict[key_name].append(key)
        flower_dict[value_name].append(value)

    # creating pandas dataframe
    pd.set_option('display.max_rows', 500)
    df = pd.DataFrame(flower_dict)
    df.set_index(key_name, inplace=True)
    df = (df.sort_values(value_name))   
    df.reset_index(key_name, inplace=True)

    # combining all frames
    df1 = df.iloc[:26, :]
    df2 = df.iloc[26:52].reset_index(drop=True)
    df3 = df.iloc[52:78, :].reset_index(drop=True)
    df4 = df.iloc[78:102, :].reset_index(drop=True)

    display(pd.concat([df1, df2, df3, df4], axis=1).fillna(''))


# In[27]:


def select_image(cat_id):
    """
    Parameters
    ----------
    cat_id: the flower id given by the respective folder number given in the flower_data folder
    
    return selected image path with image file name
    
    Note: use this to load the image path in the predict function
    """
    image_path = os.getcwd() + '\\flower_data\\valid\\{}\\'.format(cat_id)
    images = list((os.listdir(image_path)))
    try:
        images.remove('.ipynb_checkpoints')
    except:
        pass
    i = (np.random.randint(0, len(images)))
    image_name = images[i]
    selected_image = image_path + image_name
    return selected_image


# In[28]:


def predict(image_path, model, topk=5, show=True):
    """
    Parameters
    ----------
    image_path: the selected picture path with image file name
        use select_image function
    model: the trained model or loaded trained model
    topk: the top k prediction by the greatest probability -default 5
    show: to decide if just the probs and to_classes should be returned without 
        printing or showing images -default True
    
    shows picture of flower
    prints actual flower name
    returns list of probabilities and top classes by index and name
    
    Note: the index returned in top_classes can be used in the select_image function
        to return path with picture file name of the respective flower
    """
    image, cat_id = process_image(image_path)
    
    
    # prep the images to show
    actual_flower = image[:]
    image_path = ('\\'.join(image_path.split('\\')[:-2]))
    
    
    #  continue processing and returning outputs
    image = image.unsqueeze_(0) 
    image = image.to(device)
    model.to(device)      
    model.eval()
    with torch.no_grad():
        outputs = torch.exp(model.forward(image))
        
    probs, classes = outputs.topk(topk)    
   
    probs = probs.to('cpu').numpy().reshape(-1)
    tensoridx = classes.to('cpu').numpy().reshape(-1,)
    
    # brings a dictionary of tensor index to cat id
    tensoridx_to_catid = {val: key for key, val in model.class_to_idx.items()}
    
    # brings category name from category id mapping to tensor index
    top_classes = [tensoridx_to_catid[idx] + ' - ' + str(cat_to_name[tensoridx_to_catid[idx]]) for idx in tensoridx]
    
    # showing pictures
    flower_ids = [tensoridx_to_catid[idx] for idx in tensoridx]
    first_image_path = image_path + '\\' + flower_ids[0]
    second_image_path = image_path + '\\' + flower_ids[1]
    
    # first image validation
    first_flower_filename = os.listdir(first_image_path)[0]
    if first_flower_filename == '.ipynb_checkpoints':
        first_flower_filename = os.listdir(first_image_path)[1]
    
    # second image validation
    second_flower_filename = os.listdir(second_image_path)[0]
    if second_flower_filename == '.ipynb_checkpoints':
        second_flower_filename = os.listdir(second_image_path)[1]
        
    first_image_path = first_image_path + '\\' + first_flower_filename
    second_image_path = second_image_path + '\\' + second_flower_filename
    
    first_image = process_image(first_image_path)[0]
    second_image = process_image(second_image_path)[0]
    
    accurate = ' - Correct'
    color = 'black'
    if str(cat_id) != flower_ids[0]: 
        accurate = ' - Incorrect'
        color = 'red'
    
    if show: 
        title_enhancer('Actual: ' + cat_to_name[cat_id] + accurate, size=20, color=color)
        title_enhancer('1st: Actual | 2nd: First Prediction | 3rd: Second Prediction', 14)
    if show: imshow(actual_flower, first_image, second_image)  
    
    if show: plt_bar_chart(probs, top_classes) 
    
    return list(probs), top_classes


# In[29]:


def tensor_normalizer(tensor_img):
    
    tensor_norm = tensor_img
    
    # extracting channels
    img_a = tensor_norm[0]
    img_b = tensor_norm[1]
    img_c = tensor_norm[2]
   
    # normalizing 
    img_a = (img_a - 0.485)/(0.229) 
    img_b = (img_b - 0.456)/(0.224)
    img_c = (img_c - 0.406)/(0.225)
    
    # assigning to numpy matrixes
    tensor_norm[0] = img_a
    tensor_norm[1] = img_b
    tensor_norm[2] = img_c
    
    return tensor_norm


# In[30]:


def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model
    
    Parameters
    ----------
    image_path: the path of the image with the image file name
    
    returns transformed image and category id
    """    
    cat_id = (str(image_path.split('\\')[-2]))
    
    # TODO: Process a PIL image for use in a PyTorch model    
    img = Image.open(image_path)
    
    # transformer to tensor
    data_transforms = transforms.Compose([transforms.ToTensor()])
    
    # resize
    size = 256, 256
    img_tran = img.resize(size, Image.ANTIALIAS)
    
    # center crop
    width, height = img_tran.size    
    top, down = [(height-224)/2]*2  
    left, right = [(width-224)/2]*2 
    img_tran = img_tran.crop((left, top, width-right, height-down)) 
        
    tensor_img = data_transforms(img_tran)

    
    '''    # to show normalized image
    data_transformsPIL = transforms.Compose([
                                  transforms.ToPILImage()
                                  ])
    
    data_transformsPIL(tensor_img).show()
    
    data_transformsPIL(tensor_normalizer(tensor_img)).show()'''
 
    return tensor_img, cat_id


# In[31]:


def process_image_shortcut(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model
    
    Parameters
    ----------
    image_path: the path of the image with the image file name
    
    returns transformed image and category id
    """    
    cat_id = (str(image_path.split('\\')[-2]))
    
    # TODO: Process a PIL image for use in a PyTorch model    
    im = Image.open(image_path)
    
    # TODO: Process a PIL image for use in a PyTorch model
    data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456,0.406], std=[0.229, 0.224, 0.225])])
    
    '''    data_transformsPIL = transforms.Compose([
                                  transforms.ToPILImage()
                                  ])
    data_transformsPIL(data_transforms(im)).show()'''
 
    return data_transforms(im), cat_id


# In[32]:


def pretty_output(output):
    df = pd.DataFrame({'Flowers': output[1], 'Probabilities': output[0]})
    df.Probabilities = df.Probabilities.apply(lambda x: '{:.2%}'.format(x))
    df.index = df.index + 1
    display(df)


# # Re-building Model

# In[ ]:


def model_rebuilder(checkpoint_name):
    
    def run_load(checkpoint_dict):
        model_load = getattr(torchvision.models, checkpoint_dict['model'])(pretrained=True)
        model_load.classifier = checkpoint_dict['classifier']
        model_load.load_state_dict(checkpoint_dict['state_dict'])
        model_load.class_to_idx = checkpoint_dict['class_to_idx']
        return model_load
        print('Checkpoint loaded :)')
        print('-')    
    
    try:        
        return run_load(torch.load(checkpoint_name))
    except:
        try:   
            print('The default checkpoint.pth was loaded successfully.')
            print('-')
            return run_load(torch.load('checkpoint.pth'))            
        except:
            print('No checkpoint found. ')       
            
model_load = model_rebuilder(results.checkpoint_name)


# # Looking for Category to Name File

# In[35]:


def category_to_name_validation(filename='cat_to_name.json'): 
    
    if results.cat_to_name != None: 
        filename=results.cat_to_name
        with open(filename, 'r') as handle:
            cat_to_name = json.load(handle)
            return cat_to_name
        print('The argument -cat {} was used.         Names will be shown instead of category ids'.format(results.cat_to_name))
    else:
        try:
            with open(filename, 'r') as handle:
                cat_to_name = json.load(handle)
                print('A json file named cat_to_name.json was found. Names will be shown instead of category ids.')
                return cat_to_name
        except:
            print('No {} file found. Try again'.format(filename))
            cat_to_name_filenam = input(' Do you have a category-to-flower-name-mapping json file?            \n enter filename.json or no ->')
            if cat_to_name_filenam.lower() not in ('', 'no'):
                try:
                    return category_to_name_validation(cat_to_name_filenam)
                except:
                    pass
            else:                
                print('Warning! You have opted to not show flower names. Instead, the flower category ids will be shown. n\These are the same as the folders ids in which the flower images are located.')
                return model_load.class_to_idx

cat_to_name = category_to_name_validation() 
print('-')


# # Questionnaire

# In[37]:


def device_validation():
    
    if results.device == None:    
        if torch.cuda.is_available():    
            device = input('In what device do you want to run the prediction, cuda or cpu? ->')
            if device.lower() in ['cpu', 'cuda']:
                print('Thanks! You selected to run the prediction using the {}.'.format(device))
                return device
            else:
                print('Warning! Wrong input. Choose cuda or cpu.')
                return device_validation()
        else:
            device = 'cpu'
            print('Sorry! But your device does not support GPU. Note that the prediction will run faster with a GPU.             Please, consider changing to a device with a GPU.')
    else:
        return results.device
device = device_validation()
print('-')


# In[38]:


def topk_validation():
    if results.topk == None:
        try:
            topk = int(input('How many top outputs do you want to see?             \nenter an integer -->'))
            if topk >=2 and topk <= 33:                
                print('Thanks! Your output will show {} of the top predictions.'.format(topk))
                return topk
            else:
                print('Try again! Please enter integer from 2 to 33.')
                return topk_validation()
        except:
            print('Please try again! Enter an integer.')
            return topk_validation()
    else:
        return results.topk
topk = topk_validation()


# # Predicting

# In[39]:


print('These is the List of Flower Images:')
pretty_flower_frame(cat_to_name)


# In[52]:


def show_outputs():
    
    if results.image_path == None:
    
        try:
            flower_index = int(input('From the flowers in the validation set above, which one do you want to predict?            \n enter the index number or non numeric character to exit ->'))

            if flower_index > 102:
                print('\nWarning! The flower index is out of bound. Please enter an index between 1 and 102.')
                show_outputs()

            selected_image = select_image(flower_index)  

            print()
            print('\nPredicting: ' + str(flower_index) + ' - ' + 
                  string.capwords(str(cat_to_name[str(flower_index)])))

            output = predict(selected_image, model_load, topk, show=False)

            pretty_output(output)

            continue_pred = input('\nDo you want to predict another flower name? yes or no ->')

            def continue_prediction(continue_pred):
                if continue_pred.lower() == 'yes':
                    show_outputs()
                elif continue_pred.lower() == 'no':
                    print('Ok! The application will now close.')
                else:
                    print('Try again:')
                    continue_pred = input('Do you want to predict another flower name? yes or no ->')
                    continue_prediction(continue_pred)

            continue_prediction(continue_pred)           

        except Exception as e:
            print('Good bye!')
            pass
    
    else:
        try:
            output = predict(results.image_path, model_load, topk, show=False)
            pretty_output(output)
        except:
            print('Sorry! The image path provided is not valid. But you can choose to predict an image from the list above.')
            results.image_path = None
            show_outputs()
        
show_outputs()     


# In[ ]:


# D:\GitHub\Flower_Classifier_Project\flower_data\valid\23\image_03416.jpg

