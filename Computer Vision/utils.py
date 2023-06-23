import matplotlib.pyplot as plt
import torch

def plot_acc_loss(accuracy_array, loss_array, lrs):
    fig, axs = plt.subplots(1, 3)
    # plt.plot(list(range(len(accuracy_array))), accuracy_array[:])
    axs[0].plot(list(range(len(accuracy_array))), accuracy_array[:])
    axs[1].plot(loss_array[:])
    axs[2].plot(list(range(len(lrs))), lrs[:])
    fig.show()

def cal_rnn_acc(model, datas, device, hidden):
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in datas:
            x = img.to(device)
            y_ = label.to(device)

            output = model.forward(x, hidden)
            _, output_index = torch.max(output, 1)

            total += label.size(0)
            correct += (output_index == y_).sum().float()
    return 100*correct/total

def cal_lstm_acc(model, datas, device, h0, c0):
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in datas:
            x = img.to(device)
            y_ = label.to(device)

            output= model.forward(x, h0, c0)
            _, output_index = torch.max(output, 1)

            total += label.size(0)
            correct += (output_index == y_).sum().float()
    return 100*correct/total



def cal_acc(model, datas, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in datas:
            x = img.to(device)
            y_ = label.to(device)

            output = model.forward(x)
            _, output_index = torch.max(output, 1)

            total += label.size(0)
            correct += (output_index == y_).sum().float()
    return 100*correct/total