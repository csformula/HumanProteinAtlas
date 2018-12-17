import os
import time
import torch
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class Epoch_History(object):
    ''' Tracking and ploting loss and acc for each epoch. '''
    
    def __init__(self):
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
    def plot_history(self):
        self.epochs = list(range(len(self.train_loss_history)))
            
        # plot loss history
        fig1 = plt.figure(figsize=(15,5))
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Training and Validation Loss')
        plt.plot(self.epochs, self.train_loss_history, label='Training')
        plt.plot(self.epochs, self.val_loss_history, label='Validation')
        plt.legend()
        
        # plot acc history
        fig2 = plt.figure(figsize=(15,5))
        plt.title('Acc History')
        plt.xlabel('Epoch')
        plt.ylabel('Training and Validation Acc')
        plt.plot(self.epochs, self.train_acc_history, label='Training')
        plt.plot(self.epochs, self.val_acc_history, label='Validation')
        plt.legend()   
        
# calculate macro f1 score
def macrof1(input, target):
    pred = (input>0.5).to(torch.float32).numpy()
    pred_label = np.array([[28 if pred[i,j]==0 else j for j in range(pred.shape[1])] for i in range(pred.shape[0])])
    target = target.numpy()
    tar_label = np.array([[28 if target[i,j]==0 else j for j in range(target.shape[1])] for i in range(target.shape[0])])
    
    pred_label = pred_label.reshape(1,-1).squeeze().astype(np.int32)
    tar_label = tar_label.reshape(1,-1).squeeze().astype(np.int32)
    labels = []
    for i in tar_label:
        if i!=28 and i not in labels:
            labels.append(i)
    
#     print(f'tar_label:\t{tar_label}')
#     print(f'pred_label:\t{pred_label}')
#     print(f'labels:\t{labels}')

    score = f1_score(tar_label, pred_label, labels=labels, average='macro')
    return score     

# from predictions to labels(str)
def gen_results(input):
    preds = (input>0.5).to(torch.int32)
    preds_labels = [[j for j in range(preds.shape[1]) if preds[i,j]==1] for i in range(preds.shape[0])]
    labels = [' '.join(map(lambda x: str(x), i)) for i in preds_labels]
    return labels
        

def train_model(trainloader, valloader, model, optimizer, criterion, 
                scheduler=None, epochs=50, device=torch.device('cpu'), history=Epoch_History(), 
                ckp_path='.', ckp_savestep=25):
    '''
    Args:
        trainloader(list): a list containing all train dataloaders.
        valloader(list): a list containing all val dataloaders, correspond to trainloader.
        model(object): the model(base on resnet18) to be trained.
        optimizer(object): optimizer used to update weights.
        criterion(object): a nn.Module to compute loss.
        scheduler(object, optional): scheduler for updating lr, default:None.
        epochs(int, optional): epochs in each training.       
        device(object, optional): device cpu or cuda:0
        history(object, optional): a Epoch_History object to recoding loss and acc histories.
        ckt_path(str, optional): directory where checkpoint files saved.
        ckp_savestep(int, optional): save a checkpoint after training ckp_savestep epochs
    '''
    
    start = time.time()
    loops = len(trainloader)  # training times
    loader = list(zip(trainloader, valloader))
    model = model.to(device)  
    mode = [0, 1]  # 0 for train, 1 for val
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    
    for loop in range(loops):
        start_loop = time.time()
        print(f'Loop {loop+1}/{loops} processing...')
        for epoch in range(epochs):
            print(f'epoch {epoch+1}...', end=' ')
            for m in mode:
                if m==0:
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_acc = 0.0
                # train and validation at the same time
                for i, samples in enumerate(loader[loop][m]):
                    imgs, labels = samples[0].to(device), samples[1].to(device)
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(m==0):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                        acc = macrof1(outputs, labels)
                        if m==0:
                            loss.backward()
                            optimizer.step()
                            
                    running_loss += loss.item()
                    running_acc += acc
                
                avg_loss = running_loss/len(loader[loop][m])
                avg_acc = running_acc/len(loader[loop][m])  
                if m == 0:
                    history.train_loss_history.append(avg_loss)
                    history.train_acc_history.append(avg_acc)
                else:
                    history.val_loss_history.append(avg_loss)
                    history.val_acc_history.append(avg_acc)
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        best_state = copy.deepcopy(model.state_dict())
            # update lr
            if scheduler is not None:
                scheduler.step(running_loss)
                            
            print('done')
            
            # save checkpoints every 25 epochs
            if (epoch+1)%ckp_savestep == 0:
                print(f'saving checkpoint-loop{loop+1}_epoch{epoch+1}...', end=" ")
                if not os.path.exists(ckp_path):
                    os.mkdir(ckp_path) 
                torch.save({
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'acc_history': {'train': history.train_acc_history,
                                            'val': history.val_acc_history},
                            'loss_history':{'train': history.train_loss_history,
                                            'val': history.val_loss_history}
                           }, os.path.join(ckp_path, f'checkpoint-loop{loop+1}_epoch{epoch+1}.ckpt'))
                print('done')
        
        end_loop = time.time()
        print(f'Loop {loop+1}/{loops} finished! Time used: {(end_loop-start_loop):.1f}s')
        print(f'loss: {history.val_loss_history[-1]:.2f}, acc: {history.val_acc_history[-1]:.3f}')
        print('-'*30)
    
    torch.save(best_state, './weights.pt')
    end = time.time()
    print(f'Total time used: {(end-start):.1f}s')
    
    return history
                            


def test_model(testloader, model, device=torch.device('cpu'), results_filename='0.csv'):
    '''
    Inference testset and save the results into a csv file.
    Args:
        testloader(object): a dataloader containing all test images.
        model(object): the trained model(base on resnet18) for testing.     
        device(object, optional): device cpu or cuda:0.
        results_name(str, optional): file name of final results csv.
    '''
    
    print(f'Testing ...')
    start = time.time()
    model = model.to(device)
    model.eval()
    ids = []
    labels = []
    
    for i, samples in enumerate(testloader):
        batch_ids, imgs = samples[0], samples[1].to(device)
        
        with torch.no_grad():
            outputs = model(imgs)
            preds = (outputs>0.5).to(torch.int32)
            batch_labels = gen_results(preds)
        
        for n in range(len(batch_ids)):
            ids.append(batch_ids[n])
            labels.append(batch_labels[n])
        
        for k in [1,2,3]:
            if i==len(testloader)//4*k:
                print(f'{k*25:2d}% done ...')
        
    results_dict = {'Id': ids,
                    'Predicted': labels}
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(results_filename, index=False)
    
    end = time.time()
    print(f'Test Finished! Time used: {(end-start):<.1f}s')
    
    return results_dict
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
