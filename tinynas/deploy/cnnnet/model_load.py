# Copyright (c) Alibaba, Inc. and its affiliates.

from tqdm import tqdm
import ast
import os
import argparse
import torch
from cnnnet import CnnNet
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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

def load_model(model,
               load_parameters_from,
               strict_load=False,
               map_location=torch.device('cpu'),
               **kwargs):
    if not os.path.isfile(load_parameters_from):
        raise ValueError('bad checkpoint to load %s' % (load_parameters_from))
    else:
        model.logger.debug('Zennas: loading params from '
                           + load_parameters_from)
    checkpoint = torch.load(load_parameters_from, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # print("\n#################################")
    # for name, paramets in model.named_parameters():
    # print(name, paramets.size(), paramets.flatten().cpu().detach().numpy()[0:5])
    #     if "conv_offset.weight" in name:
    #         state_dict[name] = state_dict[name.replace(".conv_offset", "")]
    model.load_state_dict(state_dict, strict=strict_load)

    # print("\n#################################")
    # for name, paramets in model.named_parameters():
    #     print(name, paramets.size(), paramets.flatten().cpu().detach().numpy()[0:5])

    return model

if __name__ == '__main__':
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # make input
    args = parse_args()

    # model_get
    model, network_arch = get_model(args.structure_txt, 
                                            args.pretrained)
        
    model.load_state_dict(torch.load('./tinynas/deploy/cnnnet/tinynascnnex1_statedict_1.pth'))

    # GPU 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # state load

    # dataset
    val_dataset = datasets.ImageFolder('/home/shared/DATA/imagenet/val', transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total} %')
        


    







