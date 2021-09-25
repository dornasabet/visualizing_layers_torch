import torch
from torchvision import transforms
from PIL import Image
from models import vgg_cbam_extended
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary


def plt_show(features_, scale=None):
    plt.figure(figsize=(20, 20))
    if len(features_.shape) == 4:
        features_ = features_[0]
    if type(features_) is not np.ndarray:
        features_ = np.array(features_)
    # features_ = features_.transpose((1,2,0))
    w, h, n_channles = features_.shape
    if scale:
        features_ = cv2.resize(features_, (w * scale, h * scale))
    print(features_.shape)
    square = math.ceil(math.sqrt(n_channles))
    for ix in range(1, n_channles + 1):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features_[:, :, ix - 1], cmap='gray')
    plt.show()


class GetFeatures(nn.Module):
    def __init__(self, input, model, features, device='cuda'):
        super(GetFeatures, self).__init__()
        self.features = features
        self.input = input
        self.model = model
        self.device = device
        self.selected_out = dict()
        self.forward_hooks = []
        self.apply_hooks(model)

    def apply_hooks(self, model):
        def forward_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(self.selected_out)
                m_key = "%s-%i" % (class_name, module_idx + 1)
                self.selected_out[str(m_key)] = output

            if (not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)):
                self.forward_hooks.append(module.register_forward_hook(hook))

        self.selected_out = OrderedDict()
        model.apply(forward_hook)

    def forward(self, x):
        self.selected_out = dict()
        output = self.model(x)
        self.selected_out = {key: val for en, (key, val) in enumerate(self.selected_out.items(), 1) if
                             en in self.features}
        return output, self.selected_out


if __name__ == "__main__":

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(40),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    img = Image.open('img.jpg')
    torch_img = data_transforms['val'](img)

    model = vgg_cbam_extended.VggCBAM(cbam_blocks=(0, 1, 2, 3, 4), residual_cbam=True).to('cuda')
    summary(model, (1, 40, 40))
    model = model.to('cpu')
    # model1 = nn.ModuleList(model.modules())
    feature = [1, 12]
    new_model = GetFeatures(torch_img, model, feature)

    output, output_dict = new_model(torch.unsqueeze(torch_img, dim=0))

    # plt_show(features)
    for i, j in output_dict.items():
        print(i, j.size())
