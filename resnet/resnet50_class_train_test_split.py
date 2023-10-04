import random
import os 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import pandas as pd
from torch.utils.data import Dataset

from sklearn.model_selection import GroupShuffleSplit

# 디바이스 설정 (GPU 사용 가능하면 GPU 사용하도록)
device = torch.device("cuda:2" )

random_seed = 30

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

np.random.seed(random_seed)
random.seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 데이터 transform 적용하기 

classa = 'class13_no' 

transform = transforms.Compose([
      
      
    transforms.ToTensor(), 
    transforms.RandomHorizontalFlip(),   
    transforms.RandomVerticalFlip(),     
    transforms.RandomRotation(30),
     # 이미지를 10% 만큼 랜덤하게 이동
      # 50% 확률로 이미지에 원근 변환 적용 
    
])

transform_valid = transforms.Compose([
    transforms.ToTensor(), 
           
    
])

# 사용자 정의 Dataset 클래스
class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 2]
        image = np.load(img_path).astype(np.float32),    # npy 파일을 로드합니다.

        if isinstance(image, tuple):
            image = image[0]
            
        
        
        
        if self.transform:
           
            
            image = self.transform(image)
            

        return image, int(label)

class StratifiedGroupShuffleSplit:
    def __init__(self, n_splits=1, test_size=None, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)

    def split(self, X, y, groups):
        unique_labels = np.unique(y)
        train_indices, test_indices = [], []
        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            label_groups = groups.iloc[label_indices]
            for train_idx, test_idx in self.gss.split(X.iloc[label_indices], y.iloc[label_indices], label_groups):
                train_indices.append(label_indices[train_idx])
                test_indices.append(label_indices[test_idx])
        return np.concatenate(train_indices), np.concatenate(test_indices)








# 데이터 불러오기
total_dataset = pd.read_csv(f'/home/minkyoon/crohn/csv/clam/clustering/relapse_resnet/class_united/{classa}_clustering.csv')

# StratifiedGroupShuffleSplit 초기화
sgss = StratifiedGroupShuffleSplit(n_splits=1, train_size=0.8, random_state=random_seed)

# Train 데이터의 인덱스를 반환
train_idx, other_idx = sgss.split(total_dataset, total_dataset['label'], total_dataset['accession_number'])

# 남은 데이터에서 valid와 test를 나눕니다. (1:1 비율)
valid_and_test_data = total_dataset.iloc[other_idx]
sgss_valid_test = StratifiedGroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_seed)

valid_idx, test_idx = sgss_valid_test.split(valid_and_test_data, valid_and_test_data['label'], valid_and_test_data['accession_number'])

# 인덱스를 사용해 데이터를 분할합니다.
train_data = total_dataset.iloc[train_idx]
valid_data = valid_and_test_data.iloc[valid_idx]
test_data = valid_and_test_data.iloc[test_idx]

# 각 데이터셋에 대해 CustomImageDataset 인스턴스를 생성하고, 각각 다른 transform을 적용합니다.
train_dataset = CustomImageDataset(train_data, transform=transform)
valid_dataset = CustomImageDataset(valid_data, transform=transform_valid)
test_dataset = CustomImageDataset(test_data, transform=transform_valid)

# 이후에 DataLoader 생성 과정은 이전과 동일합니다.

    

# DataLoader을 위한 hyperparameter 설정

train_params = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 1,
    'drop_last': False}

    #num workers?

valid_params = {
    'batch_size': 64,
    'shuffle': False,
    'num_workers': 1,
    'drop_last': False}



train_loader = torch.utils.data.DataLoader(dataset=train_dataset, **train_params)
valid_loader =torch.utils.data.DataLoader(dataset=valid_dataset, **valid_params)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, **valid_params)





# dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                              batch_size=4, shuffle=True,
 #                                             num_workers=4)



# Train DataLoader 데이터 확인해보기

for x, y in train_loader:
    print(x.shape)
    print(y.shape)
    break

# Valid DataLoader 데이터 확인해보기

for x, y in valid_loader:
    print(x.shape)
    print(y.shape)
    break

"""# 모델 만들기"""

from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 모델 설정 값

config = {
    # Classfier 설정
    "cls_hidden_dims" : [1024, 512, 256]
    }


class ResNet(nn.Module):
    """pretrain 된 ResNet을 이
    """
    
    def __init__(self):
        """
		Args:
			base_model : resnet18 / resnet50
			config: 모델 설정 값
		"""
        super(ResNet, self).__init__()
       
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        self.num_ftrs = num_ftrs
        
        
        for name, param in model.named_parameters():
            if 'layer2' in name:
                break
            param.requires_grad = False

            

        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        b = x.size(0)
        x = x.view(b, -1)

        return x





model_image = ResNet()
model_image

class Classifier(nn.Sequential):
    """임베딩 된 feature를 이용해 classificaion
    """
    def __init__(self, model_image, **config):
        """
        Args:
            model_image : image emedding 모델
            config: 모델 설정 값
        """
        super(Classifier, self).__init__()

        self.model_image = model_image # image 임베딩 모델

        self.input_dim = model_image.num_ftrs # image feature 사이즈
        self.dropout = nn.Dropout(0.5) # dropout 적용

        self.hidden_dims = config['cls_hidden_dims'] # classifier hidden dimensions
        layer_size = len(self.hidden_dims) + 1 # hidden layer 개수
        dims = [self.input_dim] + self.hidden_dims + [2] 

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)]) # classifer layers 

    def forward(self, v):
        # Drug/protein 임베딩
        v_i = self.model_image(v) # batch_size x hidden_dim 

        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor)-1):
                # If last layer,
                v_i = l(v_i)
            else:
                # If Not last layer, dropout과 ReLU 적용
                v_i = F.relu(self.dropout(l(v_i)))

        return v_i

model = Classifier(model_image, **config)
model



"""# 모델 학습 (Model training)

### 모델 학습을 위한 설정
"""



# 학습 진행에 필요한 hyperparameter 

learning_rate = 0.00001
train_epoch   = 100

# optimizer 
weight_decay = 0.001  
opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

loss_fn = torch.nn.CrossEntropyLoss()

import copy
from prettytable import PrettyTable
from time import time
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

"""### 모델 학습 진횅"""




loss_history_train = []
loss_history_val = []

max_acc = 0

# 모델 GPU 메모리에 올리기
model = model.to(device)

# Best 모델 초기화
model_best = copy.deepcopy(model)

# 결과 정리를 위한 PrettyTable
valid_metric_record = []
valid_metric_header = ["# epoch"] 
valid_metric_header.extend(["Accuracy", "sensitivity", "specificity", "roc_score"])
table = PrettyTable(valid_metric_header)

float2str = lambda x:'%0.4f'%x # float 소숫점 4자리까지만 str로 바꾸기 

x
for epo in range(train_epoch):
    # Model training 
    model.train()
    
    epoch_train_loss=0
    n_batches_train = 0
    epoch_val_loss = 0
    n_batches_val = 0
    
    # Mini-batch 학습 
    for i, (v_i, label) in enumerate(train_loader):
        # input data gpu에 올리기 
        v_i = v_i.float().to(device) 
        # forward-pass
        output = model(v_i) 

        # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
        loss = loss_fn(output, label.to(device))

        # 각 iteration 마다 loss 기록 
        
        
        
        
  

        # gradient 초기화
        opt.zero_grad()
        # back propagation
        loss.backward()
        # parameter update
        opt.step()
        epoch_train_loss += loss.item()
        n_batches_train += 1
    loss_history_train.append(epoch_train_loss / n_batches_train)
    # gradient tracking X
    with torch.set_grad_enabled(False):
        
        y_pred = []
        y_score = []
        y_label = []
        # model validation
        model.eval()

        for i, (v_i, label) in enumerate(valid_loader):
            # validation 입력 데이터 gpu에 올리기
            v_i = v_i.float().to(device)

            # forward-pass
            output = model(v_i)

            # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
            loss = loss_fn(output, label.to(device))

            # 각 iteration 마다 loss 기록 
            

            pred = output.argmax(dim=1, keepdim=True)
            score = nn.Softmax(dim = 1)(output)[:,1]

            # 예측값, 참값 cpu로 옮기고 numpy 형으로 변환
            pred = pred.cpu().numpy()
            score = score.cpu().numpy()
            label = label.cpu().numpy()

            # 예측값, 참값 기록하기
            y_label = y_label + label.flatten().tolist()
            y_pred = y_pred + pred.flatten().tolist()
            y_score = y_score + score.flatten().tolist()
            epoch_val_loss += loss.item()
            n_batches_val += 1
        loss_history_val.append(epoch_val_loss / n_batches_val)
    
    # metric 계산
    classification_metrics = classification_report(y_label, y_pred,
                        target_names = ['0', '1'],
                        output_dict= True)
    
    # sensitivity is the recall of the positive class
    sensitivity = classification_metrics['0']['recall']
    # specificity is the recall of the negative class 
    specificity = classification_metrics['1']['recall']
    # accuracy
    accuracy = classification_metrics['accuracy']
    # confusion matrix
    conf_matrix = confusion_matrix(y_label, y_pred)
    # roc score
    roc_score = roc_auc_score(y_label, y_score)

    # 계산한 metric 합치기
    lst = ["epoch " + str(epo)] + list(map(float2str,[accuracy, sensitivity, specificity, roc_score]))

    # 각 epoch 마다 결과값 pretty table에 기록
    table.add_row(lst)
    valid_metric_record.append(lst)
    
    # mse 기준으로 best model 업데이트
    if accuracy > max_acc:
        # best model deepcopy 
        # model_best = copy.deepcopy(model)

        best_model_wts = copy.deepcopy(model.state_dict())
        # max MSE 업데이트 
        max_acc = accuracy

    

    # 각 epoch 마다 결과 출력 
    print('Validation at Epoch '+ str(epo + 1) + ' , Accuracy: ' + str(accuracy)[:7] + ' , sensitivity: '\
						 + str(sensitivity)[:7] + ', specificity: ' + str(f"{specificity}") +' , roc_score: '+str(roc_score)[:7])


best_model = model.load_state_dict(best_model_wts)
model_best.load_state_dict(best_model_wts)
torch.save(model_best, f'{classa}_resnet50_pi.pt')
import matplotlib.pyplot as plt

# 학습 곡선 그리기
def plot_loss_curve(loss_history_train, loss_history_val, save_path):
    plt.figure(figsize=(10,7))
    plt.plot(loss_history_train, label='Train')
    plt.plot(loss_history_val, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # 이미지 저장
    plt.close()
    
#저장경로
    
# ... training code ...

# 학습이 끝난 후에 학습 곡선을 그립니다.
# 이미지 저장 경로 설정
save_path = f'{classa}_loss_curv_pi.png'  # 원하는 경로와 파일명으로 변경하세요.
plot_loss_curve(loss_history_train, loss_history_val, save_path)

# """# 모델 테스트 (model testing)"""

# # Test dataloader 확인 
for i, (v_i, label) in enumerate(test_loader):
    print(v_i.shape)
    print(label.shape)
    break

# """### 모델 테스트 진행"""

# # 테스트 진행

# model = model_best

y_pred = []
y_label = []
y_score = []



model = torch.load(f'{classa}_resnet50_pi.pt')
model.eval()
for i, (v_i, label) in enumerate(test_loader):
    # input data gpu에 올리기 
    v_i = v_i.float().to(device)

    with torch.set_grad_enabled(False):
        # forward-pass
        output = model(v_i)

        # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
        loss = loss_fn(output, label.to(device))

        # 각 iteration 마다 loss 기록 

        pred = output.argmax(dim=1, keepdim=True)
        score = nn.Softmax(dim = 1)(output)[:,1]

        # 예측값, 참값 cpu로 옮기고 numpy 형으로 변환
        pred = pred.cpu().numpy()
        score = score.cpu().numpy()
        label = label.cpu().numpy()

    # 예측값, 참값 기록하기
    y_label = y_label + label.flatten().tolist()
    y_pred = y_pred + pred.flatten().tolist()
    y_score = y_score + score.flatten().tolist()

# # metric 계산
classification_metrics = classification_report(y_label, y_pred,
                    target_names = ['0', '1'],
                    output_dict= True)
# sensitivity is the recall of the positive class
sensitivity = classification_metrics['0']['recall']
# specificity is the recall of the negative class 
specificity = classification_metrics['1']['recall']
# accuracy
accuracy = classification_metrics['accuracy']
# confusion matrix
conf_matrix = confusion_matrix(y_label, y_pred)
# roc score
roc_score = roc_auc_score(y_label, y_score)

# 각 epoch 마다 결과 출력 


print('Validation, Accuracy: ' + str(accuracy)[:7] + ', Sensitivity: ' 
      + str(sensitivity)[:7] + ', Specificity: ' + str(f"{specificity}")[:7] 
      + ', ROC Score: ' + str(roc_score)[:7])

# """### 테스트 결과 시각화"""

# plot the roc curve    
fpr, tpr, _ = roc_curve(y_label, y_score)
plt.plot(fpr, tpr, label = "Area under ROC = {:.4f}".format(roc_score))
plt.legend(loc = 'best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig(f'{classa}_roc_pi.png')
plt.show()
plt.close()

import seaborn as sns

conf_matrix = conf_matrix
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, fmt='d',ax = ax, cmap = 'Blues'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
plt.savefig(f'{classa}_confusition_pi.png')
plt.close()

result_string = ('Validation, Accuracy: ' + str(accuracy)[:7] + ', Sensitivity: ' 
      + str(sensitivity)[:7] + ', Specificity: ' + str(f"{specificity}")[:7] 
      + ', ROC Score: ' + str(roc_score)[:7]+"class1_class2인경우 class1에서 relapse라벨뽑은거임!")
with open(f'{classa}_results_pi.txt', 'w') as f:
    f.write(result_string)


