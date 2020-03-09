import torch
import torchvision
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:23].eval())

        for bl in blocks:
            bl.to(device)
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to(device)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to(device)
        self.resize = resize

    def forward(self, out_unet,image,ind):
        if out_unet.shape[1] != 3:
            out_unet = out_unet.repeat(1, 3, 1, 1)
            image = image.repeat(1, 3, 1, 1)
#         import ipdb;ipdb.set_trace()
        out_unet = (out_unet-self.mean) / self.std
        image = (image-self.mean) / self.std
        if self.resize:
            out_unet = self.transform(out_unet, mode='bilinear', size=(224, 224), align_corners=False)
            image = self.transform(image, mode='bilinear', size=(224, 224), align_corners=False)
        x = out_unet
        y = image
        
        x = self.blocks[ind](x)
        y = self.blocks[ind](y)
        loss = torch.nn.functional.l1_loss(x, y)
        return loss