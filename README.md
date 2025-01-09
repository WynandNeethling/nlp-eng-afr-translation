# nlp-eng-afr-translation
Optimising English-to-Afrikaans Translation with RNNs and Attention Mechanisms

### Overview
This repository contains a single Jupyter Notebook file that includes all the code for the project.
The notebook covers data loading, preprocessing, model building, training, and evaluation.

### Dependencies
To run the code in the notebook, you need to have the following dependencies installed:

from io import open
import os
import re
import random
import xml.etree.ElementTree as ET

import gensim.downloader as api

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Running the Code
To run the code, simply open the Jupyter Notebook file and execute the cells sequentially. 
The notebook is structured to guide you through the entire process, from data loading to model evaluation

### Device Configuration
The code automatically detects if a CUDA-enabled GPU is available and uses it for training. 
If not, it defaults to using the CPU.
