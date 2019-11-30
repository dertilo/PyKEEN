#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import pykeen
from pykeen.kge_models import ConvE


# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


logging.basicConfig(level=logging.INFO)
logging.getLogger('pykeen').setLevel(logging.INFO)


# print(sys.version)

# In[4]:


print(time.asctime())


# In[5]:


print(pykeen.get_version())


# Check which hyper-parameters are required by ConvE:

# In[6]:


ConvE.hyper_params


# Define output directory:

# In[7]:


output_directory = os.path.join(
    os.path.expanduser('~'), 
    'Desktop', 
    'pykeen_test'
)


# Define hyper-parameters:

# Note: ConvE_height * ConvE_width == embedding_dim

# Note: ConvE_kernel_height <= ConvE_height

# Note: ConvE_kernel_width <= ConvE_width

# In[10]:


config = dict(
    training_set_path           = 'tests/resources/data/rdf.nt',
    execution_mode              = 'Training_mode',
    random_seed                 = 0,
    kg_embedding_model_name     = 'ConvE',
    embedding_dim               = 50,
    ConvE_input_channels        = 1,  
    ConvE_output_channels       = 3,  
    ConvE_height                = 5,
    ConvE_width                 = 10,
    ConvE_kernel_height         = 5,
    ConvE_kernel_width          = 3,
    conv_e_input_dropout        = 0.2,
    conv_e_feature_map_dropout  = 0.5,
    conv_e_output_dropout       = 0.5,
    margin_loss                 = 1,
    learning_rate               = 0.01,
    num_epochs                  = 20,  
    batch_size                  = 64,
    preferred_device            = 'cpu'
)


# Train ConvE:

# In[11]:


results = pykeen.run(
    config=config,
    output_directory=output_directory,
)


# Check result entries:

# In[12]:


results.results.keys()


# Access trained model:

# In[13]:


results.results['trained_model']


# Visualize loss values:

# In[14]:


losses = results.results['losses']
epochs = np.arange(len(losses))
plt.title(r'Loss Per Epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(epochs, losses)
plt.show()

