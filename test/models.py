import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module): 
    def __init__(self, in_features):#
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  
            nn.Conv2d(in_features, in_features, 3), 
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),  
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)#64 256 64 64 


class GeneratorResNet1(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):    #1 25 25 , 9
        super(GeneratorResNet1, self).__init__()

        channels = input_shape[0]  #  1

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),  #镜像填充1 27 27
            nn.Conv2d(channels, out_features, 3),  #64 25 25
            nn.InstanceNorm2d(out_features),  #一种BN层
            nn.ReLU(inplace=True),
        ]
        in_features = out_features  #64

        # Residual blocks  #64 25 25
        for _ in range(num_residual_blocks): #9个參差模块 维度大小不变 64 25 25
            model += [ResidualBlock(out_features)]


        # Output layer  
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape  # 1 25 25

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1 ,5 ,5 )  

        def discriminator_block(in_filters, out_filters, normalize=True):  #1 25 25
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding = 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def discriminator_block1(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers



        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),  #64 25 25
            *discriminator_block(64, 128),  #128 25 25


            *discriminator_block1(128, 256),  #256 12 12
            *discriminator_block1(256, 512),  #512 5 5
            nn.Conv2d(512, 1, 3, padding=1)  #1 5 5
        )

    def forward(self, img):
        return self.model(img)





















