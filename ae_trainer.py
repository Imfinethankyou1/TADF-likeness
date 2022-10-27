import time
import torch
import torch.optim as optim
import numpy as np


class AETrainer():

    def __init__(self, ae_net, dataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
            batch_size: int = 128, weight_decay: float = 1e-6,  save_model: str = 'trained_model/save_model.pt', device: str = 'cuda', lr_decay: float = 0.99):
        self.save_model = save_model
        ae_net = ae_net.to(device)
        self.ae_net = ae_net
        self.dataset = dataset
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestones = lr_milestones
        self.n_epochs = n_epochs
        self.optimizer_name = optimizer_name
        self.lr_decay = lr_decay
        self.c = None
        test_idx_list = []
        test_inputs_list = []
        labels_list = []
        if dataset is not None:
            train_loader = self.dataset[0]
            test_loader = self.dataset[1]
            if test_loader is not None:
                for data in test_loader:
                    inputs = data['fp']
                    labels = data['target']
                    idx = data['key']
                    if isinstance(idx,list):
                        idx = [int(i)  for i in idx]
                        idx = torch.Tensor(idx)
                        idx = idx.to(self.device)
                    inputs = inputs.to(self.device)
                    inputs = inputs.float()
                    test_inputs_list.append(inputs)
                    test_idx_list.append(idx)
                    labels_list.append(labels)
                self.test_inputs_list = test_inputs_list
                self.test_idx_list = test_idx_list
                self.labels_list = labels_list
            inputs_list = []
            for data in train_loader:
                inputs = data['fp']
                inputs = inputs.to(self.device)
                inputs = inputs.float()
                inputs_list.append(inputs)
            self.inputs_list = inputs_list

    def train(self):

        ae_net = self.ae_net
        
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)

        # Training
        print('Training on')
        start_time = time.time()
        

        min_loss = 100
        count = 0 
        for epoch in range(self.n_epochs):
            scheduler.step()
            #if epoch in self.lr_milestones:
                #    print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            ae_net.train()
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for inputs in self.inputs_list:
                # Zero the network parameter gradients
                optimizer.zero_grad()
                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            if self.dataset[1] is not None:
                test_loss = self.test()
            else:
                test_loss = loss_epoch/n_batches
            if test_loss < min_loss:
                min_loss = test_loss
                torch.save(ae_net.state_dict(), self.save_model)
                count = 0
            else:
                count +=1
                if count > 1000:
                    break
                ### Save model prameter setting
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f} Test Loss : {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, test_loss))

        ae_net.load_state_dict(torch.load(self.save_model))
        pretrain_time = time.time() - start_time

        return ae_net

    def test(self):

        # Set device for network
        ae_net = self.ae_net

        # Testing
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        with torch.no_grad():
            ae_net.eval()
            for idx, inputs in zip(self.test_idx_list, self.test_inputs_list):
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                loss_epoch += loss.item()
                n_batches += 1

        loss= loss_epoch / n_batches
        return loss


    def SVDD_train(self):
        
        net = self.ae_net.encoder


        # Get train data loader
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        self.save_model = 'trained_model/hypersphere.pt'
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            self.c = self.init_center_c()

        # Training
        print('Starting SVDD training...')
        start_time = time.time()
        net.train()

        min_loss_epoch = np.float('inf')
        min_cnt = 0
        for epoch in range(self.n_epochs):

            scheduler.step()

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for inputs in self.inputs_list:

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            val_loss=self.SVDD_test()
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f} Test Loss : {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, val_loss))

            if val_loss < min_loss_epoch:
                min_loss_epoch = val_loss
                torch.save(net.state_dict(), self.save_model)
                min_cnt = 0
            else:
                min_cnt += 1
                if min_cnt > 5:
                    break

        net.load_state_dict(torch.load(self.save_model))
        self.train_time = time.time() - start_time
        self.c = self.c.cpu().data.numpy().tolist()
        print('Finished SVDD training.')

        return net
    def SVDD_test(self):
        net = self.ae_net.encoder
        val_loss_epoch = 0.0
        val_batches = 0
        #for data in test_loader:
        for idx, inputs in zip(self.test_idx_list, self.test_inputs_list):
            with torch.no_grad():
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                loss = torch.mean(dist)

                val_loss_epoch += loss.item()
                val_batches +=1
        loss = val_loss_epoch/val_batches
        return loss

#logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f} Val loss : {:.8f}'
            #            .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, val_loss_epoch/val_batches))


    def init_center_c(self, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        net = self.ae_net.encoder
        train_loader = self.inputs_list
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for inputs in train_loader:
                # get the inputs of the batch
                #inputs, _, _ = data
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

