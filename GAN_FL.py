
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
import pandas
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split
from baseline import *

import sys
import torch
import numpy as np


from utils.models import FashionMNISTCNN,AttackGenerator
from utils.basics import generic_train, test_total_accuracy, test_class_accuracy, save_model
from utils.attacks import NoAttack, RandomAttack, TargetedAttack, UAPAttack,GANAttack
from utils.defenses import NoDefense, FlippedLabelsDefense
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"








if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    batch_size = 32
    lr = 1e-3
    num_common_layer = 16
    num_common_layer1= 4
    num_common_layer2 = 0
    num_epochs = 1
    num_clients = 10
    rounds =30
    verbose = True
    # Threat Model
    malicious_upscale = 20  # Scale factor for parameters update
    num_malicious = 3
    attack = GANAttack()

    defense = FlippedLabelsDefense(num_classes=1)



    federated_baseline = FederatedBaseline(num_clients=num_clients, device=device, num_common_layer = num_common_layer)
    federated_baseline.load_data(batch_size, num_clients)
    federated_baseline.configure_attack(attack=attack, num_malicious=num_malicious)
    federated_baseline.configure_defense(defense=defense)

    print(federated_baseline.train(
        num_epochs=num_epochs,
        rounds=rounds,
        lr=lr,
        num_common_layer=num_common_layer,
        malicious_upscale=malicious_upscale,
        verbose=verbose))
    print(federated_baseline.test(0))
    federated_baseline.model = FashionMNISTCNN()

    print(federated_baseline.train(
        num_epochs=num_epochs,
        rounds=rounds,
        lr=lr,
        num_common_layer=num_common_layer1,
        malicious_upscale=malicious_upscale,
        verbose=verbose))

    print(federated_baseline.test(0))
    federated_baseline.model = FashionMNISTCNN()

    print(federated_baseline.train(
        num_epochs=num_epochs,
        rounds=rounds,
        lr=lr,
        num_common_layer=num_common_layer2,
        malicious_upscale=malicious_upscale,
        verbose=verbose))

    print(federated_baseline.test(0))

