```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```



# 链接kaggle

```python
!pip install -U -q kaggle
!mkdir -p ~/.kaggle
!echo '{"username":"ys1995","key":"55186b9952feb72e14df109705c229e6"}' > ~/.kaggle/kaggle.json
# {"username":"ys1995","key":"55186b9952feb72e14df109705c229e6"} 在kaggle账号里生成
!chmod 600 ~/.kaggle/kaggle.json
!mkdir -p data
!kaggle competitions download -c digit-recognizer -p data
# 404 进行重启colab
```



# 数据集读取csv

```python
class MNIST_data(Dataset):
    """MNIST dtaa set"""
    # transforms.ToTensor()自动除以255
    
    def __init__(self, file_path, 
                 transform = transforms.Compose([transforms.ToTensor(), 
                     transforms.Normalize(mean=(0.5,), std=(0.5,))])
                ):
        
        df = pd.read_csv(file_path)
        
        if len(df.columns) == n_pixels:
            # test data
            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = None
        else:
            # training data
            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = torch.from_numpy(df.iloc[:,0].values)
            
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])
```



# 测试集读取时一定不能打乱顺序

```python
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, shuffle=False)
```

# 测试集要进行和训练集相同的变化，保证数据分布的相同



# 参数初始化

```python
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
```

# 学习率调整

```python
# 等间隔调整学习率，调整倍数为 gamma 倍，调整间隔为 step_size。间隔单位是step。需要注意的是， step 通常是指 epoch，不要弄成 iteration 了。


# torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)

scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
```

# 损失函数的选择

是使用torch自带的交叉熵函数`nn.CrossEntropyLoss()`，还是` F.log_softmax(output, dim=1)`与`loss = F.nll_loss(output, target)`的组合,target都是数字，而不是对应得onehot向量

# 保存最好的模型

```python
import copy
# 有学习率的调整,保存最好的模型
def train_loop(epochs, model, optimizer, scheduler, criterion, device, dataloader):
    model = model.to(device)
    model.train()
    loss_hist, acc_hist = [], []
    best_acc = 0.
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        since = time.time()
        running_loss = 0.
        running_correct = 0
        scheduler.step()
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # 感觉不用with也行
            with torch.set_grad_enabled(True):
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # torch.max(input, dim, keepdim=False, out=None) 
                # -> (Tensor, LongTensor)
				# 按维度dim 返回最大值
                # torch.max(a,1) 返回每一行中最大值的那个元素，
                # 且返回其索引（返回最大元素在这一行的列索引）
                
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_correct += torch.sum(preds == labels.detach())
        
        # len(dataloader.dataset)才是总的数据样本量
        # len(dataloader) = 样本总量//batch
        epoch_loss = running_loss / len(dataloader.dataset)
        # 计算一次epoch的平均loss
        epoch_acc = running_correct.item() / len(dataloader.dataset)
        # 计算一次epoch的准确率
        
        # 保存loss和acc便于绘图
        loss_hist.append(epoch_loss)
        acc_hist.append(epoch_acc)
        
        # 保存最好的模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Epoch: {} / {}, Loss: {:.4f}, Accuracy:{:.4f}, Time: {:.0f}m {:.0f}s'.format(
            epoch + 1, epochs, epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy: {:.4f}'.format(best_acc))
    return best_model_wts, loss_hist, acc_hist


epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') # 1m 44s
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)


model_state_dict, loss_hist, acc_hist = \
    train_loop(epochs, model, optimizer, scheduler, criterion, device, train_loader)
```



```python
import time
def eval_loop(model, device, dataloader):
    model.to(device)
    model.eval()
    result = None
    since = time.time()
    for images in dataloader:
        since = time.time()
#         visualize_example(images)
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
#             print(preds.cpu().numpy()[:64])
            if result is None:
                result = preds.cpu().numpy().copy()
            else:
                result = np.hstack((result, preds.cpu().numpy()))
    time_elapsed = time.time() - since
    print('Time: {:.0f}m {:.0f}s {:.0f}ms'.format(
        time_elapsed // 60, time_elapsed % 60, time_elapsed * 1000 % 1000))
    return result
# 加载最好的模型
model.load_state_dict(model_state_dict)
result = eval_loop(model, device, test_loader)


image_id = np.arange(1, len(result) + 1)
result_np = np.hstack((image_id.reshape((-1, 1)), result.reshape(-1, 1)))
result_df = pd.DataFrame(result_np, columns=['ImageId', 'Label'])
# print(result_df.head(64))
result_df.to_csv('cnn_pytorch_result3.csv', index=False)
```

