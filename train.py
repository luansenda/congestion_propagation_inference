import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from gcns import GCNS1,GCNS2,GCNS3,GCNS4
from dbgcn_utils import generate_dataset, load_data, get_normalized_adj

use_gpu = False
num_timesteps_input = 1
num_timesteps_output = 1

epochs = 20
batch_size = 36

parser = argparse.ArgumentParser(description='DBGCN')
parser.add_argument('--enable_cuda', action='store_true', help='Enable CUDA')
parser.add_argument('--data',type=str,default='data/node_values16.npy',help='data path')
parser.add_argument('--adjdata',type=str,default='data/adj_mat16_lea.npy',help='adj data path')
parser.add_argument('--adj_lea', action='store_true', help='using the learned adj_matrix')
parser.add_argument('--save', type=str, default='checkpoints/stg',help='model save')
args = parser.parse_args()
args.device = None
args.mat_flag = False

if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

if args.adj_lea:
    args.mat_flag = True
else:
    args.mat_flag = False


def train_epoch(training_input, training_target, batch_size):
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        if(i%50==0):
            print("Iter {:03d}: train loss is {:.4f}".format(i,epoch_training_losses[-1]))
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(7)

    A, X = load_data(args.data, args.adjdata)

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)
    print("*********** Data load successfully! *********")

    A_wave = get_normalized_adj(A, args.mat_flag)
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(device=args.device)
    
    ## Try using a different GCN structure, GCNS1 or GCNS2 or GCNS3 or GCNS4
    net = GCNS1(A_wave.shape[0],
               training_input.shape[3],
               num_timesteps_input,
               num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    validation_mapes = []
    validation_rmses = []
    for epoch in range(epochs):
        ## training
        loss = train_epoch(training_input, training_target, batch_size=batch_size)
        training_losses.append(loss)
        ## validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)
            out = net(A_wave, val_input)
            val_loss = loss_criterion(out, val_target).to(device="cpu")
            validation_losses.append(np.asscalar(val_loss.detach().numpy()))

            out_unnormalized = out.detach().cpu().numpy()
            target_unnormalized = val_target.detach().cpu().numpy()

            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            mape = np.mean(np.absolute(out_unnormalized - target_unnormalized)/target_unnormalized)
            rmse = np.sqrt(np.mean((out_unnormalized - target_unnormalized)**2))
            validation_maes.append(mae)
            validation_mapes.append(mape)
            validation_rmses.append(rmse)

            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        print("Epoch {:03d}--Training loss: {:.4f}".format(epoch+1, training_losses[-1]))
        print("Epoch {:03d}--Validation loss: {:.4f}".format(epoch+1, validation_losses[-1]))
        print("Epoch {:03d}--Validation MAE: {:.4f}--Validation MAPE: {:.4f}--Validation RMSE: {:.4f}".format(epoch+1, validation_maes[-1], validation_mapes[-1], validation_rmses[-1]))
        torch.save(net.state_dict(),args.save + "_epoch_" + str(epoch) + ".pth")

    ## plot train & validation losses
    plt.plot(training_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.legend()
    plt.show()

    ## testing
    print("*************** Testing .... *****************")
    bestid = np.argmin(validation_losses)
    net.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid)+".pth"))

    with torch.no_grad():
        net.eval()
        test_input = test_input.to(device=args.device)
        test_target = test_target.to(device=args.device)
        out = net(A_wave, test_input)
        test_loss = loss_criterion(out, test_target).to(device="cpu")
        test_loss = np.asscalar(test_loss.detach().numpy())

        out_unnormalized = out.detach().cpu().numpy()
        target_unnormalized = test_target.detach().cpu().numpy()

        test_mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
        test_mape = np.mean(np.absolute(out_unnormalized - target_unnormalized) / target_unnormalized)
        test_rmse = np.sqrt(np.mean((out_unnormalized - target_unnormalized) ** 2))

        out = None
        test_input = val_input.to(device="cpu")
        test_target = val_target.to(device="cpu")

    print("Testing finished")
    print("The test loss on best model is", str(round(test_loss,4)))
    print("--Testing MAE: {:.4f} --Testing MAPE: {:.4f} --Testing RMSE: {:.4f}".format(test_mae, test_mape, test_rmse))

    print("Groundtruth size:",target_unnormalized.shape)
    print("Predictions size:", out_unnormalized.shape)
    ## plot prediction & groundtruth -- road-1
    plt.plot(target_unnormalized[:,0,0], label="Groundtruth")
    plt.plot(out_unnormalized[:,0,0], label="Predictions")
    plt.ylim(0,1)
    plt.legend()
    plt.show()

    ## plot prediction & groundtruth  -- road-2
    plt.plot(target_unnormalized[:,1,0], label="Groundtruth")
    plt.plot(out_unnormalized[:,1,0], label="Predictions")
    plt.ylim(0,1)
    plt.legend()
    plt.show()

    ## 转df,存储本地
    import pandas as pd
    y1 = target_unnormalized[:,0,0]
    yhat1 = out_unnormalized[:,0,0]

    y2 = target_unnormalized[:,1,0]
    yhat2 = out_unnormalized[:,1,0]
    mae2 = abs(y2-yhat2)

    y3 = target_unnormalized[:,2,0]
    yhat3 = out_unnormalized[:,2,0]
    mae3 = abs(y3-yhat3)

    y4 = target_unnormalized[:,3,0]
    yhat4 = out_unnormalized[:,3,0]
    mae4 = abs(y4-yhat4)

    y5 = target_unnormalized[:,4,0]
    yhat5 = out_unnormalized[:,4,0]
    mae5 = abs(y5-yhat5)

    y6 = target_unnormalized[:,5,0]
    yhat6 = out_unnormalized[:,5,0]
    mae6 = abs(y6-yhat6)

    y7 = target_unnormalized[:,6,0]
    yhat7 = out_unnormalized[:,6,0]
    mae7 = abs(y7-yhat7)
    y8 = target_unnormalized[:,7,0]
    yhat8 = out_unnormalized[:,7,0]
    mae8 = abs(y8-yhat8)

    y9 = target_unnormalized[:,8,0]
    yhat9 = out_unnormalized[:,8,0]
    mae9 = abs(y9-yhat9)

    y10 = target_unnormalized[:,9,0]
    yhat10 = out_unnormalized[:,9,0]
    mae10 = abs(y10-yhat10)

    y11 = target_unnormalized[:,10,0]
    yhat11 = out_unnormalized[:,10,0]
    mae11 = abs(y11-yhat11)
    y12 = target_unnormalized[:,11,0]
    yhat12 = out_unnormalized[:,11,0]
    mae12 = abs(y12-yhat12)

    y13 = target_unnormalized[:,12, 0]
    yhat13 = out_unnormalized[:,12, 0]
    mae13 = abs(y13 - yhat13)

    y14 = target_unnormalized[:, 13, 0]
    yhat14 = out_unnormalized[:, 13, 0]
    mae14 = abs(y14 - yhat14)

    y15 = target_unnormalized[:, 14, 0]
    yhat15 = out_unnormalized[:, 14, 0]
    mae15 = abs(y15 - yhat15)

    y16 = target_unnormalized[:, 15, 0]
    yhat16 = out_unnormalized[:, 15, 0]
    mae16 = abs(y16 - yhat16)

    df2 = pd.DataFrame({'r1':mae2,'r2':mae3, 'r3': mae4, 'r4':mae5,'r5':mae6,'r6':mae7, 'r7': mae8, 'r8':mae9,'r9':mae10,'r10':mae11, 'r11': mae12, 'r12':mae13,'r13':mae14,'r14':mae15, 'r15': mae16})
    df2.to_excel('data/net.xlsx',index=False)

'''
    ## Inference with
    plt.plot(target_unnormalized[120:150, 0, 0], label="Evidence")
    plt.plot(out_unnormalized[120:150,1,0], label="R2")
    plt.plot(out_unnormalized[120:150,2,0], label="R3")
    plt.plot(out_unnormalized[120:150, 3, 0], label="R4")
    plt.plot(out_unnormalized[120:150, 4, 0], label="R5")
    plt.plot(out_unnormalized[120:150, 5, 0], label="R6")
    plt.plot(out_unnormalized[120:150, 6, 0], label="R7")
    plt.plot(out_unnormalized[120:150, 7, 0], label="R8")
    plt.plot(out_unnormalized[120:150, 8, 0], label="R9")
    plt.plot(out_unnormalized[120:150, 9, 0], label="R10")
    plt.ylim(0,1)
    plt.legend()
    plt.show()
'''
