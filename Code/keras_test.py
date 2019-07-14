# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:05:23 2019

@author: Mayank Jain
"""

from keras.layers import Input, Dense, Add
from keras.models import Model
from keras.utils import plot_model

A11 = Input(shape=(30,),name='A11')
A12 = Input(shape=(30,),name='A12')
A1 = Add()([A11, A12])
A2 = Dense(8, activation='relu',name='A2')(A1)
A3 = Dense(30, activation='relu',name='A3')(A2)

B2 = Dense(40, activation='relu',name='B2')(A2)
B3 = Dense(30, activation='relu',name='B3')(B2)

merged = Model(inputs=[A11, A12],outputs=[A3,B3])
plot_model(merged,to_file='demo.png',show_shapes=True)