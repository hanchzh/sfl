# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import sys
import torch
import numpy as np
import pandas as pd
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from utils.models import FashionMNISTCNN,AttackGenerator
from utils.basics import GAN_train, generic_train, test_total_accuracy, test_class_accuracy, save_model
from utils.attacks import NoAttack, RandomAttack, TargetedAttack, UAPAttack,GANAttack
from utils.defenses import NoDefense, FlippedLabelsDefense
from utils.sampling import mnist_noniid_train_test,mnist_noniid

torch.manual_seed(1) #Set seed 
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
class Baseline:
    def __init__(self, device="cpu"):
        """
        Baseline parent class
        Args:
            device (str, optional): where to run pytorch on. Defaults to "cpu".
        """        
        self.device = torch.device(device)
        self.model = FashionMNISTCNN()
        self.model.to(self.device)


    def load_data(self, batch_size=32, num_clients=10):
        """
        load FashionMNIST data
        Args:
            batch_size (int, optional): the batch size. Defaults to 32.
        """
        self.batch_size = batch_size
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,), )])  # normalize to [-1,1]
        self.trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        self.testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        self.trainsets = []
        self.testsets = []
        self.stds = []
        self.trainloaders = []
        self.testloaders = []
        num_items = int(len(self.trainset) / num_clients)
        dict_users, all_idxs = {}, [i for i in range(len(self.trainset))]
        for i in range(num_clients):
            std = 0.5-(i-num_clients/2.0)/100
            #std = 0.5
            self.stds.append(std)
            self.batch_size = batch_size
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((std,), (std,), )])  # normalize to [-1,1]
            trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            self.trainloaders.append(torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True))
            self.testloaders.append(torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False))
            self.trainsets.append(trainset)
            self.testsets.append(testset)
            dict_users[i] = set(np.random.choice(all_idxs, int(num_items/400), replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])

        self.dict_users = dict_users


        return dict_users
    def test(self,i):
        """
        test the accuracy of the model
        Returns:
            (tuple[float]): the overall and class-wise accuracies of the model
        """        
        total_acc = test_total_accuracy(self.model, self.testloaders[i], self.device)
        class_acc = test_class_accuracy(self.model, self.testloaders[i], self.device)
        return total_acc, class_acc


    def _make_optimizer_and_loss(self, lr, momentum=0.9):
        """
        helper function to create an optimizer and loss function
        Args:
            lr (float): the learning rate
            momentum (float, optional): the momentum. Defaults to 0.9.
        Returns:
            (tuple[torch.nn.CrossEntropyLoss, torch.optim.SGD]): criterion and optimizer functions
        """        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        return criterion, optimizer



class GAN(Baseline):
    def __init__(self, device="cpu",num_common_layer = -1):
        """
        Basic CNN Baseline
        Args:
            device (str, optional): the device to run pytorch on. Defaults to "cpu".
        """        
        super(GAN, self).__init__(device=device)
        self.num_common_layer = num_common_layer

    def set_trainloader(self, trainloader):
        """
        set the trainloader for the model
        Args:
            trainloader (torch.utils.data.Dataloader): the training data data loader
        """        
        self.trainloader = trainloader


    def configure_attack(self, attack=NoAttack()):
        """
        configure malicious attacks on the model
        Args:
            attack (Attack, optional): The attack to apply to the model. Defaults to NoAttack().
        """
        self.attack = attack


    def train(self, round, num_epochs, lr=1e-3, verbose=False, print_summary=True):
        """
        train the basic baseline model
        Args:
            num_epochs (int): the number of epochs.
            lr (float, optional): the learning rate. Defaults to 1e-3.
            verbose (bool, optional): do you want print output? Defaults to False.
            print_summary (bool, optional): print the hyperparameters. Defaults to True.
        Returns:
            (list[floats]): the training losses
        """  

        if print_summary:
            print(f"Training BasicBaseline model.")	            
            print("========== HYPERPARAMETERS ==========")
            print(f"round: {round}")
            print(f"num_epochs: {num_epochs}")	            
            print(f"lr: {lr}")	
            print(f"attack: {self.attack}")
            print("\n")

        criterion, optimizer = self._make_optimizer_and_loss(lr)
        gan_optimizer = optim.Adam(self.attack.generator.parameters(), lr=lr)
        return GAN_train(
            model=self.model,
            round=round,
            num_epochs=num_epochs,
            num_common_layer = self.num_common_layer,
            trainloader=self.trainloader, 
            optimizer=optimizer,
            g_optimizer=gan_optimizer,
            criterion=criterion,  
            attack=self.attack,
            device=self.device, 
            verbose=verbose)


class BasicBaseline(Baseline):
    def __init__(self, device="cpu"):
        """
        Basic CNN Baseline
        Args:
            device (str, optional): the device to run pytorch on. Defaults to "cpu".
        """
        super(BasicBaseline, self).__init__(device=device)

    def set_trainloader(self, trainloader):
        """
        set the trainloader for the model
        Args:
            trainloader (torch.utils.data.Dataloader): the training data data loader
        """
        self.trainloader = trainloader

    def configure_attack(self, attack=NoAttack()):
        """
        configure malicious attacks on the model
        Args:
            attack (Attack, optional): The attack to apply to the model. Defaults to NoAttack().
        """
        self.attack = attack

    def train(self, round, num_epochs, lr=1e-3, verbose=False, print_summary=True):
        """
        train the basic baseline model
        Args:
            num_epochs (int): the number of epochs.
            lr (float, optional): the learning rate. Defaults to 1e-3.
            verbose (bool, optional): do you want print output? Defaults to False.
            print_summary (bool, optional): print the hyperparameters. Defaults to True.
        Returns:
            (list[floats]): the training losses
        """

        if print_summary:
            print(f"Training BasicBaseline model.")
            print("========== HYPERPARAMETERS ==========")
            print(f"round: {round}")
            print(f"num_epochs: {num_epochs}")
            print(f"lr: {lr}")
            print(f"attack: {self.attack}")
            print("\n")

        criterion, optimizer = self._make_optimizer_and_loss(lr)
        return generic_train(
            model=self.model,
            rounds=round,
            num_epochs=num_epochs,
            trainloader=self.trainloader,
            optimizer=optimizer,
            criterion=criterion,
            attack=self.attack,
            device=self.device,
            verbose=verbose)

class FederatedBaseline(Baseline):
    def __init__(self, num_clients, device="cpu", num_common_layer = -1):
        """
        Federated CNN baseline model
        Args:
            num_clients (int): number of clients for federated learning
            device (str, optional): where to run pytorch on. Defaults to "cpu".
        """        
        super(FederatedBaseline, self).__init__(device=device)
        self.num_clients = num_clients
        self.round_log = []
        self.num_commom_layer = num_common_layer


    def configure_attack(self, attack=NoAttack(), num_malicious=0):
        """
        configure malicious attacks against the model from clients
        Args:
            attack (Attack, optional): the attack type. Defaults to NoAttack().
            num_malicious (int, optional): number of malicious clients using this attack. Defaults to 0.
        """        
        assert num_malicious <= self.num_clients, "num_malicious must be <= num_clients"
        self.attack = attack
        self.num_malicious = num_malicious
        self.attacks = [attack for i in range(num_malicious)]
        self.attacks.extend([NoAttack() for i in range(self.num_clients - num_malicious)])

    
    def manual_attack(self, attack_list):
        """
        manually set the attacks
        Args:
            attack_list (iterable[Attack]): the attacks
        """        
        assert len(attack_list) == self.num_clients, "len(attack_list) must be == num_clients"
        self.attacks = attack_list


    def configure_defense(self, defense):
        """
        configure the federated learning defense
        Args:
            defense (Defense): the defense
        """        
        self.defense = defense


    def train(self, num_epochs, rounds=1, lr=1e-3,num_common_layer=-1, malicious_upscale=1.0, log=True, verbose=False, print_summary=True):
        """
        train the federated baseline model
        Args:
            num_epochs (int): the number of epochs
            rounds (int, optional): the number of rounds to train clients. Defaults to 1.
            lr (float, optional): the learning rate. Defaults to 1e-3.
            malicious_upscale (float, optional): scale factor for parameter updates of the malicious models.
            log (boolean, optional): to log the round-wise accuracies. Defaults to True.
            verbose (bool, optional): do you want print output? Defaults to False.
            print_summary (bool, optional): print the hyperparameters. Defaults to True.
        Returns:
            (list[floats]): the training losses
        """   

        if print_summary:
            print(f"Training FederatedBaseline model with {self.num_clients} clients.")	            
            print("========== HYPERPARAMETERS ==========")
            print(f"num_clients: {self.num_clients}")
            print(f"num_epochs: {num_epochs}")
            print(f"num_epochs: {num_common_layer}")
            print(f"rounds: {rounds}")	            
            print(f"lr: {lr}")	
            print(f"num_malicious: {self.num_malicious}")
            print(f"attack: {self.attack}")
            print(f"malicious_upscale: {malicious_upscale}")
            print(f"defense: {self.defense}")
            print(f"log: {log}")
            print("\n")         

        train_losses = []
        clients = []
        client_trainloaders = self._make_client_trainloaders()
        for i in range(self.num_clients):
            if i == 11:
                client = GAN(device=self.device,num_common_layer = num_common_layer)
                client.set_trainloader(client_trainloaders[i])
                client.configure_attack(attack=self.attacks[i])

            else:
                client = BasicBaseline(device=self.device)
                client.set_trainloader(client_trainloaders[i])
                client.configure_attack(attack=self.attacks[i])
            clients.append(client)
        for r in range(rounds):

            round_loss = 0.0
            client_models = []
            for i in range(self.num_clients):

                client = clients[i]
                client.model.load_state_dict(self.sfed(self.model.state_dict(), client.model, num_common_layer))
                loss = client.train(
                    num_epochs=num_epochs,
                    round=r,
                    lr=lr,
                    verbose=verbose,
                    print_summary=False
                )[-1]
                client_models.append(client.model.state_dict())
                round_loss += loss
                # if verbose:
                #     print(f"--> client {i} trained, round {r} \t final loss: {round(loss, 3)}\n")

            train_losses.append(round_loss / self.num_clients)
            #self._aggregate(client_models, malicious_upscale)
            self.model.load_state_dict(self.sfedAvg(client_models,num_common_layer))
            if log:
                accuracies = self.test(0)
                overall, classwise = accuracies
                total = classwise.tolist()
                total.insert(0, overall)
                self.round_log.append(total)
        self.model.load_state_dict(client_models[0])
        return train_losses


    def _make_client_trainloaders(self):
        """
        helper function to create client trainloader splits
        Returns:
            (list[torch.utils.data.Dataloader]): a list of dataloaders for the split data
        """
        data_user = []
        for i in self.dict_users:
            data_user.append(DataLoader(DatasetSplit(self.trainsets[i], self.dict_users[i]), batch_size=self.batch_size, shuffle=True))
        return data_user
    def _aggregate(self, client_models, malicious_upscale):
        """
        global parameter updates aggregation.
        Args:
            client_models (list[torch.nn.Module]): the client models
            malicious_upscale (float): scale factor for parameter updates
        """    
        ### take simple mean of the weights of models ###
        safe_clients = self.defense.run(self.model, client_models, plot_name="fig.png")
        global_dict = self.model.state_dict()
        for k in global_dict.keys():
            update = [safe_clients[i].state_dict()[k].float() for i in range(len(safe_clients))]
            update[:self.num_malicious] *= malicious_upscale
            global_dict[k] = torch.stack(update, axis=0).mean(axis=0)
        self.model.load_state_dict(global_dict)
            

    def sfed(self, w, net_user,num_common_layer):
        w_users = []
        w_user = net_user.state_dict()
        for k in list(w_user.keys())[:num_common_layer]:
            w_user[k] = w[k]
        return w_user
    def sfedAvg(self, w, num_common_layer):
        w_avg = copy.deepcopy(w[0])
        for k in list(w_avg.keys())[:num_common_layer]:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg








