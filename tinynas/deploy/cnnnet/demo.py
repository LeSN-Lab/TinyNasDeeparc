# Copyright (c) Alibaba, Inc. and its affiliates.

from tqdm import tqdm
import ast
import os
import argparse
import torch
from cnnnet import CnnNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure_txt', type=str)
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()
    return args

def get_model(filename,
                pretrained=True,
                network_id=0,
                classification=True #False for detetion
                 ):
    # load best structures
    with open(filename, 'r') as fin:
        content = fin.read()
        output_structures = ast.literal_eval(content)

    network_arch = output_structures['space_arch']
    best_structures = output_structures['best_structures']

    # If task type is classification, param num_classes is required
    out_indices = (1, 2, 3, 4) if not classification else (4, )
    model = CnnNet(
            structure_info=best_structures[network_id],
            out_indices=out_indices,
            num_classes=1000,
            classification=classification)
    model.init_weights(pretrained)

    return model, network_arch


if __name__ == '__main__':
    # make input

    args = parse_args()

    x = torch.randn(1, 3, 224, 224)

    # instantiation
    model, network_arch = get_model(args.structure_txt, 
                                            args.pretrained)

    #print(model)
    # forward
    input_data = [x]
        
    # model eval & pred
    model.eval()
    pred = model(*input_data)

    #print output
    for o in pred:
        print(o.size())

    import torch
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader

    # 데이터 전처리
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder('/home/shared/DATA/imagenet/train', transform=transform_train)
    val_dataset = datasets.ImageFolder('/home/shared/DATA/imagenet/val', transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    import torch.optim as optim
    import torch.nn as nn

    # 모델, 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # GPU 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 학습 루프
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs) #TODO:얘문제

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

        # 평가 루프
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Validation Accuracy: {100 * correct / total} %')

    # torch save
    try:
        torch.save(model.state_dict(), 'tinynascnnex1_statedict_10.pth')
    except Exception as e:
        print("model.save()에서 오류가 발생했습니다.")