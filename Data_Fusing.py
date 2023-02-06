import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataloader import DataLoader
from Blender import AutoEncoder

torch.set_num_threads(1)


def main(args):
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

  train_dataset = torch.load("Filename of the (label, image, sentence) training tuples.")
  test_dataset = torch.load("Filename of the (label, image, sentence) testing tuples.")

  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batchsize)
  test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batchsize)

  if args.train == "yes":
    ae = AutoEncoder().to(device)
    
    if args.cvextractor=="alexnet":
      feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True).to(device)

    if args.cvextractor=="resnet18":
      feature_extractor = models.resnet18(pretrained=True).to(device)

    if args.cvextractor=="mobilenetv2":
      feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).to(device)

    if args.cvextractor=="vgg11":
      feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).to(device)

    # Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=1e-5)

    for epoch in range(args.epochs):
      loss = 0
      for labels, cvs, nlps in train_loader:
        labels = labels.to(device)
        cvs = cvs.to(device)
        nlps = nlps.to(device)

        nlps = torch.unsqueeze(nlps, 1)

        # update parameters
        optimizer.zero_grad()
          
        fused, nlp_features = ae(cvs, nlps)

        cv_features = feature_extractor(fused)

        feature_loss = criterion(cv_features, nlp_features)
        vision_loss = criterion(fused, cvs)
        train_loss = feature_loss + vision_loss
        
        train_loss.backward()
          
        optimizer.step()
          
        loss += train_loss.item()
      
      loss = loss / len(train_loader)
      print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, args.epochs, loss))

    torch.save(ae, "Filename of the Blender.")

  # Evaluation
  else:
    ae = torch.load("Filename of the Blender.").to(device)
  
  ae.eval()

  train_hijacking_list = []
  test_hijacking_list = []
  
  train_idx = 0
  test_idx = 0

  with torch.no_grad():
    for labels, cvs, nlps in train_loader:
      labels = labels.to(device)
      cvs = cvs.to(device)
      nlps = nlps.to(device)

      nlps = torch.unsqueeze(nlps, 1)

      fused, _ = ae(cvs, nlps)
      
      for i in range(len(labels)):
        train_hijacking_list.append([])
        train_hijacking_list[train_idx].append(int(labels[i]))
        train_hijacking_list[train_idx].append(fused[i].cpu())
        
        train_idx += 1

  torch.save(train_hijacking_list, "Filename of the fused training dataset.")

  with torch.no_grad():
    for labels, cvs, nlps in test_loader:
      labels = labels.to(device)
      cvs = cvs.to(device)
      nlps = nlps.to(device)

      nlps = torch.unsqueeze(nlps, 1)

      fused, _ = ae(cvs, nlps)

      for i in range(len(labels)):
        test_hijacking_list.append([])
        test_hijacking_list[test_idx].append(int(labels[i]))
        test_hijacking_list[test_idx].append(fused[i].cpu())

        test_idx += 1

  torch.save(test_hijacking_list, "Filename of the fused testing dataset.")
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fusing sentences and images.")
  parser.add_argument("--gpu", default=6, type=int, help="Decide to run the program on which gpu")
  parser.add_argument("--cvextractor", default="vgg11", type=str, choices=["mobilenetv2", "vgg11"])
  parser.add_argument("--container", default="cifar10", type=str, choices=["cifar10", "mnist", "stl10"])
  parser.add_argument("--nlpextractor", default="bert", type=str, choices=["bert", "bart"])
  parser.add_argument("--hijacking", default="yelp", type=str, choices=["yelp", "sogou"])
  parser.add_argument("--epochs", default=200, type=int, help="Number of epochs")
  parser.add_argument("--train", default="yes", type=str, choices=["yes", "no"])
  parser.add_argument("--batchsize", default=64, type=int, help="The value of batch size")
  args = parser.parse_args()

  main(args)