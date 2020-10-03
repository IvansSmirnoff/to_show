#!/usr/bin/env python
# coding: utf-8

# In[103]:


# Библиотеки: нужные и ненужные

import tarfile
import matplotlib.pyplot as plt
import time
import json
import copy
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
import torchvision
from torch.utils.data import Dataset
import os
from natsort import natsorted
import numpy as np
import json
import glob


# расположение папки с фотографиями
train_dir = input()

# Класс датасет для считывания данных с папки
class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

#трансформер для тестовой выборки
data_transforms = transforms.Compose([
    transforms.Resize((224,224), interpolation=2), # Приводим к одному размеру
    transforms.Grayscale(num_output_channels=3),  # Приводим к серому цвету- для пола не нужен цвет кожи.
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
])

# Передача данных из папки в loader
my_dataset = CustomDataSet(train_dir, transform=data_transforms)
test_loader = torch.utils.data.DataLoader(my_dataset , batch_size=len(my_dataset), shuffle=False, 
                               num_workers=0)


#Загружаем модель
model = torch.load('saved_model')
# Добавляем в конце слой softmax для вывода класса в виде вероятностей
model = nn.Sequential(
    model,
    nn.Softmax(1)
)

# Модель предсказывает:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                    
for inputs in test_loader:
    inputs = inputs.to(device)
    outputs = model(inputs) 

# Список с именами фотографий
filenames = glob.glob("faces/*.*")

# Работать с numpy привычней
solution = outputs.cpu().detach().numpy()

# Создание словаря с ответами
process_results = {}
process_results['gender_results'] = []
for number in range(len(solution)):
    if solution[number, 0] > solution[number, 1]:
        process_results['gender_results'].append({
            filenames[number]: 'female'})
    else:
        process_results['gender_results'].append({
            filenames[number]: 'male'})
        
with open('process_results.json', 'w') as fp:
    json.dump(process_results, fp)

print('лох!')

