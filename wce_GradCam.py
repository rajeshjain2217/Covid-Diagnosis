import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
import torch.nn.functional as F
import torch.nn.modules.linear as ln
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
device=torch.device("mps")
data_dir=('project 8 DATA by  TA/kvasir labelled_images/kvasir _labelled_images/')
# image_path=data_dir+'/'

# image_paths=[]
# image_labels=[]
# num_images=100
# for label in os.listdir(image_path):
#     if(label=='.DS_Store'):
#       continue
#     print('\t',label)
#     img_count=len(os.listdir(image_path+label+"/"))
#     num_images=img_count
#     if(num_images>1000):
#       num_images=1000
#     for image in os.listdir(image_path+label+"/"):
#       if(num_images>0):
#         print('\t\t',image_path+label+'/'+image,label)

#         image_paths.append(image_path+label+'/'+image)
#         image_labels.append(label)
#         num_images=num_images-1
# import pandas as pd

# # Convert list to dictionary
# data_dict = {'image_path':image_paths  , 'label':image_labels}

# # Convert dictionary to pandas DataFrame
# df = pd.DataFrame.from_dict(data_dict)

# # Save DataFrame to CSV file
# csv_file_path = 'wce_data.csv'
# df.to_csv(csv_file_path)
# print("CSV file has been created successfully.")

wce_data=pd.read_csv("../wce_data.csv")
print(wce_data.head())
print("stastics of the given data:")
print(wce_data['label'].value_counts())
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
wce_data["en_label"]=lb.fit_transform(wce_data["label"])
print(wce_data.tail())
# # sx=sns.countplot(x="label",data=wce_data)
# # labels=sx.get_xticklabels()
# # sx.set_xticklabels(labels=labels)
# #plt.show()
import numpy as np
x_train,x_test,y_train,y_test=train_test_split(wce_data["image_path"].to_numpy(),wce_data["en_label"].to_numpy(),test_size=0.2,random_state=42)
print(len(x_train))
print(len(x_test))
transform=transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])
from PIL import Image
class Data(Dataset):
  def __init__(self,X,Y):
    self.X=X
    self.Y=Y
    self.len=len(self.X)

  def __getitem__(self,index):
    X=Image.open("../"+self.X[index]).convert("RGB")
    return transform(X),self.Y[index]

  def __len__(self):
    return self.len
train=Data(x_train,y_train)
test=Data(x_test,y_test)
# batch_size=5
# trainloader=DataLoader(train,batch_size=batch_size,shuffle=True)
# testloader=DataLoader(test,batch_size=batch_size,shuffle=False)
# #device=torch.device("cuda:0")
# model_compare=pd.DataFrame()
# model_loss=pd.DataFrame()
# import torchvision
# from torchvision import models
# import torch.optim as optim
# def compare_model(model_name):
#   loss_arr=[]
#   if(model_name=="resnet"):
#     model=models.resnet18(pretrained=True)
#   elif(model_name=="alexnet"):
#     model=models.alexnet(pretrained=True)
#   elif(model_name=="densenet"):
#     model=models.densenet121(pretrained=True)
#   elif(model_name=="vgg"):
#     model=models.vgg11_bn(pretrained=True)
#   elif(model_name=="squeezenet"):
#     model=models.squeezenet1_0(pretrained=True)
#   model=model.to(device)
#   criterion=nn.CrossEntropyLoss()
#   optimizer=optim.SGD(model.parameters(),lr=0.0001)
#   num_epochs=50
#   for epoch in tqdm (range(num_epochs)):
#     for image,label in trainloader:
#       image=image.to(device)
#       label=label.to(device)
#       optimizer.zero_grad()
#       outputs=model(image)
#       loss=criterion(outputs,label)
#       loss.backward()
#       optimizer.step()
#     loss_arr.append(loss.item())
#   model_loss[model_name]=loss_arr
#   total=0
#   correct=0
#   for images,labels in testloader:
#       images=images.to(device)
#       labels=labels.to(device)
#       optimizer.zero_grad()
#       output = model(images)
#       _,out=torch.max(output,1)
#       correct+=torch.sum(out==labels.data)
#       total+=len(labels)
#       optimizer.step()
#   accuracy=correct/total
#   print(model_name,": ",accuracy)
#   torch.save(model.state_dict(),'wce_model.pt')

#compare_model("resnet")
#compare_model("alexnet")
#compare_model("densenet")
#compare_model("vgg")
#compare_model("squeezenet")

# Accuracy resnet(88.83), alexnet(90.34,88.89), densenet(89.19), vgg(90.94), squeezenet(90.70)
# total runtime(61.28 hrs)= resnet(6.3 hrs), alexnet(2.51 hrs,16.58 min), densenet(24.4 hrs), vgg(25.03 hrs), squeezenet(3.04 hrs)


import torchvision
from torchvision import models
import cv2
path='wce_model.pt'
model=models.densenet121(pretrained=True)
model.load_state_dict(torch.load(path))
model.eval()
binimgs=[]
# print(test[0][0])
# #print(model)
from torchcam.methods import GradCAMpp,GradCAM
with GradCAM(model, target_layer="denselayer16", input_shape=(3,100,100)) as cam_extractor:
# # # # Preprocess your data and feed it to the mopel
  out = model(test[0][0].unsqueeze[0])
# # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
# # activation _map = cam_extractor(torch.maxout.data, 1)[1]. data.cpu() -numpy (), out)
  binimg=activation_map[0].squeeze(0)
  binimg=cv2.resize(binimg.cpu().numpy(), (100,100), interpolation=cv2.INTER_LINEAR)
  binimgs.append(torch.from_numpy(binimg).unsqueeze(dim=0))
binimgs = torch.cat(binimgs, dim=0).to(device)
numpy_binimgs=binimgs.squeeze(0).cpu().numpy()
numpy_binimgs=numpy_binimgs*255
numpy_binimgs = numpy_binimgs.astype(np.uint8)
colormap = cv2.applycolorMap(numpy_binimgs, cv2.COLORMAP_VIRIDIS)
imgs1=Image.open('../'+x_test[0]).convert("RGB")
numpy_imgs1 = imgs1.squeeze(0).permute(1,2,0).cpu().numpy()
numpy_imgs1=numpy_imgs1*255
numpy_imgs1 = numpy_imgs1.astype(np.uint8)
overlay = cv2.addweighted(numpy_imgs1, 0.7, colormap, 0.3,0)
plt.imshow(overlay)