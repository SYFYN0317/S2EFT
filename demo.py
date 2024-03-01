import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from vit_pytorch import ViT
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Trento', 'Augsburg', 'Houston'], default='Trento', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=1, help='number of evaluation')
parser.add_argument('--patches', type=int, default=7, help='number of patches')
parser.add_argument('--band_patches', type=int, default=3, help='number of related band')
parser.add_argument('--epoches', type=int, default=600, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=7):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=7, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("**************************************************")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, num_classes):
    y_train = []
    y_test = []


    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("**************************************************")
    return y_train, y_test
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()       

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)
        
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        
    return tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data
if args.dataset == 'Trento':
    folder_data = './Trento/'
    data_HS = loadmat(folder_data + 'HSI.mat')
    data_DSM1 = loadmat(folder_data + 'LiDAR.mat')
    label_TR = loadmat(folder_data + 'TRLabel.mat')
    label_TE = loadmat(folder_data + 'TSLabel.mat')

    input_HS = data_HS['HSI']
    input_DSM1 = np.expand_dims(data_DSM1['LiDAR'], axis=-1)
    TR = label_TR['TRLabel']
    TE = label_TE['TSLabel']
    height, width, band1 = input_HS.shape
    _, _, band2 = input_DSM1.shape
    input = input_HS
    # input = np.concatenate((input_HS, input_DSM1), axis=2)
    band_MultiModal = [band1, band2]
    band = band1
elif args.dataset == 'Augsburg':
    folder_data = './Augsburg/'
    data_HS = loadmat(folder_data + 'HSI.mat')
    data_DSM1 = loadmat(folder_data + 'LiDAR.mat')
    label_TR = loadmat(folder_data + 'TRLabel.mat')
    label_TE = loadmat(folder_data + 'TSLabel.mat')

    input_HS = data_HS['HSI']
    input_DSM1 = np.expand_dims(data_DSM1['LiDAR'], axis=-1)
    TR = label_TR['TRLabel']
    TE = label_TE['TSLabel']
    height, width, band1 = input_HS.shape
    _, _, band2 = input_DSM1.shape
    input = input_HS
    # input = np.concatenate((input_HS, input_DSM1), axis=2)
    band_MultiModal = [band1, band2]
    band = band1
elif args.dataset == 'Houston':

    folder_data = './Augsburg/'
    data_HS = loadmat(folder_data + 'HSI.mat')
    data_DSM1 = loadmat(folder_data + 'LiDAR.mat')
    label_TR = loadmat(folder_data + 'TRLabel.mat')
    label_TE = loadmat(folder_data + 'TSLabel.mat')

    input_HS = data_HS['HSI']
    input_DSM1 = np.expand_dims(data_DSM1['LiDAR'], axis=-1)
    TR = label_TR['TRLabel']
    TE = label_TE['TSLabel']
    height, width, band1 = input_HS.shape
    _, _, band2 = input_DSM1.shape
    input = input_HS
    # input = np.concatenate((input_HS, input_DSM1), axis=2)
    band_MultiModal = [band1, band2]
    band = band1
else:
    raise ValueError("Unkknow dataset")
color_mat = loadmat('./data/AVIRIS_colormap.mat')
# TR = data['TR']
# TE = data['TE']
# input = data['input'] #(145,145,200)
label = TR + TE
num_classes = np.max(TR)

color_mat_list = list(color_mat)
color_matrix = color_mat[color_mat_list[3]] #(17,3)
# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)

input_normalize_DSM1 = np.zeros(input_DSM1.shape)
for i in range(input_DSM1.shape[2]):
    input_max_DSM1 = np.max(input_DSM1[:,:,i])
    input_min_DSM1 = np.min(input_DSM1[:,:,i])
    input_normalize_DSM1[:,:,i] = (input_DSM1[:,:,i]-input_min_DSM1)/(input_max_DSM1-input_min_DSM1)
# data size
height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
height_DSM1, width_DSM1, band_DSM1 = input_DSM1.shape
print("height={0},width={1},band={2}".format(height_DSM1, width_DSM1, band_DSM1))
#-------------------------------------------------------------------------------
# obtain train and test data
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
mirror_image_DSM1 = mirror_hsi(height_DSM1, width_DSM1, band_DSM1, input_normalize_DSM1, patch=args.patches)
x_train_band, x_test_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
x_train_DSM1, x_test_DSM1 = train_and_test_data(mirror_image_DSM1, 1, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
x_train_band = np.concatenate((x_train_band,x_train_DSM1),axis=-1)
x_test_band = np.concatenate((x_test_band,x_test_DSM1),axis=-1)
y_train, y_test = train_and_test_label(number_train, number_test, num_classes)
#-------------------------------------------------------------------------------
# load data
x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
Label_train=Data.TensorDataset(x_train,y_train)
x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
Label_test=Data.TensorDataset(x_test,y_test)


label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)


#-------------------------------------------------------------------------------
# create model
model = ViT(
    image_size = args.patches,
    near_band = args.band_patches,
    num_patches = band,
    num_classes = num_classes,
    dim = 64,
    depth = 5,
    heads = 4,
    mlp_dim = 8,
    dropout = 0.1,
    emb_dropout = 0.1,
    mode = args.mode
)
model = model.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
#-------------------------------------------------------------------------------
if args.flag_test == 'test':
    if args.mode == 'ViT':
        model.load_state_dict(torch.load('./ViT.pt'))      
    elif (args.mode == 'CAF') & (args.patches == 1):
        model.load_state_dict(torch.load('./pixel.pt'))
    elif (args.mode == 'CAF') & (args.patches == 7):
        model.load_state_dict(torch.load('./patch.pt'))
    else:
        raise ValueError("Wrong Parameters") 
    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

    # output classification maps
    # pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    # prediction_matrix = np.zeros((height, width), dtype=float)
    # for i in range(total_pos_true.shape[0]):
    #     prediction_matrix[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
    # plt.subplot(1,1,1)
    # plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # savemat('matrix.mat',{'P':prediction_matrix, 'label':label})
elif args.flag_test == 'train':

    best_checkpoint = {"OA_TE": 0.50}
    print("start training")
    tic = time.time()
    for epoch in range(args.epoches): 


        # train model
        model.train()

        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)

        scheduler.step()

        OA_TR, AA_TR, Kappa_TR, CA_TR = output_metric(tar_t, pre_t)

        # vis.line(X=np.array([epoch]), Y=np.array([train_acc.data.cpu().numpy()]), win='train_acc', update='append', opts={'title':'Train Accuracy'})
        # vis.line(X=np.array([epoch]), Y=np.array([train_obj.data.cpu().numpy()]), win='train_obj', update='append', opts={'title':'Train Loss'})
        scheduler.step()

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):

            print("Epoch: {:03d} train_loss: {:.4f}, train_OA: {:.2f}".format(epoch + 1, train_obj, OA_TR * 100))

            model.eval()
            tar, pre = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar, pre)
            print("Epoch: {:03d} test_loss: {:.4f}, test_OA: {:.2f}, test_AA: {:.2f}, test_Kappa: {:.4f}".format(
                epoch + 1, train_obj, OA2 * 100, AA_mean2 * 100, Kappa2))

            # vis.line(X=np.array([epoch]), Y=np.array([test_acc.data.cpu().numpy()]), win='test_oa', update='append', opts={'title':'Test Overall Accuracy'})
            # vis.line(X=np.array([epoch]), Y=np.array([AA_TE*100]), win='test_aa', update='append', opts={'title':'Test Average Accuracy'})
            # vis.line(X=np.array([epoch]), Y=np.array([test_obj.data.cpu().numpy()]), win='test_obj', update='append', opts={'title':'Test Loss'})

            if OA2 * 100 > best_checkpoint['OA_TE']:
                best_checkpoint = {'epoch': epoch, 'OA_TE': OA2 * 100, 'AA_TE': AA_mean2 * 100, 'Kappa_TE': Kappa2,
                                   'CA_TE': AA2 * 100}

    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")

print("Final result:")
print(">>> The peak performance in terms of OA is achieved at epoch", best_checkpoint['epoch'])
print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(best_checkpoint['OA_TE'], best_checkpoint['AA_TE'], best_checkpoint['Kappa_TE']))
print("CA: ", best_checkpoint['CA_TE'])
print("**************************************************")
print("Parameter:")

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

print_args(vars(args))









