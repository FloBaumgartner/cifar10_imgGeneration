#!/usr/bin/env python
# coding: utf-8

# In[140]:


import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import Dataset, DataLoader

NUMSHAPES = 4
MAXRANGE = 100
RECTANGLE = 0
TRIANGLE = 1

class Shape():
    def __init__(self, start, width, height, shapetype):
        self.start = start
        self.width = width
        self.height = height
        self.shapetype = shapetype        

class GeneratorShapes():
    
    def __init__(self, res=10):
        self.res = res
        self.mingap = 5
        self.minwidth = 5
        self.maxwidth = 10
        self.minheight = 5
        self.maxheight = 20

    def generate(self):
        # input
        x = np.linspace(0, MAXRANGE, MAXRANGE*self.res)
        y = np.zeros_like(x)
        order = self._generate_shape_order()
        shapes = self._generate_objects(order)
        self._add_shapes(x, y, shapes)

        # target
        targetx = np.copy(x)
        targety = np.zeros_like(targetx)
        means = np.zeros(2)
        count = np.zeros(2)
        for shape in shapes:
            _type = RECTANGLE if shape.shapetype==RECTANGLE else TRIANGLE
            means[_type]+=shape.height
            count[_type]+=1
        means = means/count

        target_shapes = []
        for shape in shapes:
            target_shapes.append(Shape(shape.start, shape.width, means[shape.shapetype], shape.shapetype))
        self._add_shapes(targetx, targety, target_shapes)
        return x, y, targetx, targety 

    def _add_shapes(self, x, y, shapes):
        for shape in shapes:
            if shape.shapetype==RECTANGLE:
                self._add_rectangle(x, y, shape)
            else:
                self._add_triangle(x, y, shape)

    
    # Function to generate a rectangle
    def _add_rectangle(self, x, y, shape):
        end = shape.start + shape.width
        mask = (x >= shape.start) & (x <= end)
        y[mask] = shape.height

    # Function to generate a triangle
    def _add_triangle(self, x, y, shape):
        mid = shape.start + shape.width / 2
        mask_ascend = (x >= shape.start) & (x < mid)
        mask_descend = (x >= mid) & (x <= shape.start + shape.width)
        
        y[mask_ascend] = shape.height * (x[mask_ascend] - shape.start) / (mid - shape.start)
        y[mask_descend] = shape.height * (1 - (x[mask_descend] - mid) / (shape.start + shape.width - mid))

    def _generate_shape_order(self):
        order = []
        # values 0,1 correspond to rectangles, values 2,3 to triangles
        for val in np.random.permutation(np.arange(4)):            
            if val <= 1:
                # rectangle
                order.append(RECTANGLE)
            else:
                # triangle
                order.append(TRIANGLE)
        return order
    
    def _generate_objects(self, order):
        # generate the ranges where the shapes will be located including height information
        minstart = 0
        maxstart = MAXRANGE - 4*self.maxwidth - 3*self.mingap
        shapes = []
        for shapetype in order:
            start = random.uniform(minstart,maxstart)
            width = random.uniform(self.minwidth, self.maxwidth)
            height =  random.uniform(self.minheight, self.maxheight)
            shapes.append(Shape(start, width, height, shapetype))
            minstart = start + width + self.mingap
            maxstart = maxstart + self.maxwidth + self.mingap
        return shapes

class Shapes1dDataset(Dataset):

    def __init__(self, nsamples, res):
        self.size = nsamples
        x = np.zeros((nsamples, MAXRANGE*res))
        y = np.zeros((nsamples, MAXRANGE*res))
        generator = GeneratorShapes(res)
        for i in range(nsamples):
            _,_x,_,_y = generator.generate()
            x[i]= _x
            y[i]= _y
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# In[141]:


def plot_shapes(x, y):
    plt.figure(figsize=(10, 6))
    _x = np.arange(0,len(y))
    plt.plot(_x, x, label="Input", color='b')
    plt.plot(_x, y, label="Output", color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()


# In[142]:


gen = GeneratorShapes(10)
x0, y0, x, y, = gen.generate()
plot_shapes(y0,y)


# In[145]:


training_data = Shapes1dDataset(10000,10)
training_loader = DataLoader(training_data, batch_size=32, shuffle=True)


# In[147]:


_x, _y = next(iter(training_loader))
plot_shapes(_x[0],_y[0])


# In[ ]:




