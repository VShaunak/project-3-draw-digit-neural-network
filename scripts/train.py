"""Download MNIST, train MLP and CNN, and persist real test-set metrics."""
from pathlib import Path
import os,sys,time,torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from sklearn.metrics import precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'));from main import MLP,CNN,MODELS,Sessions,Metric,Base,engine,DEVICE
Base.metadata.create_all(engine);MODELS.mkdir(exist_ok=True);train=DataLoader(datasets.MNIST(ROOT/'data',train=True,download=True,transform=transforms.ToTensor()),batch_size=128,shuffle=True);test=DataLoader(datasets.MNIST(ROOT/'data',train=False,download=True,transform=transforms.ToTensor()),batch_size=256)
epochs=int(os.getenv('EPOCHS','5'))
for name,model in [('mlp',MLP()),('cnn',CNN())]:
 model.to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.001);loss=nn.CrossEntropyLoss();model.train()
 for _ in range(epochs):
  for x,y in train:opt.zero_grad();z=loss(model(x.to(DEVICE)),y.to(DEVICE));z.backward();opt.step()
 model.eval();actual=[];pred=[]
 with torch.no_grad():
  for x,y in test:pred.extend(model(x.to(DEVICE)).argmax(1).cpu().tolist());actual.extend(y.tolist())
 precision,recall,f1,_=precision_recall_fscore_support(actual,pred,average='weighted',zero_division=0);accuracy=sum(a==b for a,b in zip(actual,pred))/len(actual);torch.save(model.cpu().state_dict(),MODELS/f'{name}.pt')
 with Sessions()as s:s.merge(Metric(name=name,accuracy=accuracy,precision=precision,recall=recall,f1=f1,parameters=sum(p.numel()for p in model.parameters())));s.commit()
 print(name,accuracy)
