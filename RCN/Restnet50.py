import torch
import torch.nn as nn

def Conv1(in_planes,places,stride = 2):  #1 64
    return nn.Sequential(
        nn.Conv2d(in_channels = in_planes,out_channels = places,kernel_size = 7,stride = stride,padding = 3,bias = False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace = True),
)
'''
def Conv2(in_planes,places,stride = 2):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_planes,out_channels = places,kernel_size = 3,stride = stride,padding = 1,bias = False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace = True),

)
'''
def Conv3(in_planes,places,stride = 2):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_planes,out_channels = places,kernel_size = 3,stride = stride,padding = 1,bias = False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace = True),

)

def Conv4(in_planes,places,stride = 2):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_planes,out_channels = places,kernel_size = 9,stride = stride,bias = False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace = True),

)

def Conv5(in_planes,places,stride = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_planes,out_channels = places,kernel_size = 1,stride = stride,bias = False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace = True),

)

class Bottleneck(nn.Module):
    def __init__(self,in_planes,places,stride = 1,downsampling = False,expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels = in_planes,out_channels = places,kernel_size = 1,stride = 1,bias = False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace = True),#inplace是否覆盖
            nn.Conv2d(in_channels = places,out_channels = places,kernel_size = 3,stride = stride,padding = 1,bias = False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = places,out_channels = places * self.expansion,kernel_size = 1,stride = 1,bias = False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels = in_planes,out_channels = places * self.expansion,kernel_size = 1,stride = stride,bias = 1),
                nn.BatchNorm2d(places * expansion),
            )

        self.relu = nn.ReLU(inplace = True)

    def forward(self,x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out +=residual
        out = self.relu(out)
        return out

class RestNet(nn.Module):
    def __init__(self,block,num_classes = 1000,expansion = 4):
        super(RestNet,self).__init__()
        self.expansion = expansion

        self.conv1= Conv1(in_planes = 1,places = 64)

        self.layer1 = self.make_layer(inplaces = 64,places = 64,block = block[0],stride = 1)
        self.layer2 = self.make_layer(inplaces = 256,places = 128,block = block[1],stride = 2)
        #self.layer3 = self.make_layer(inplaces = 512,places = 256,block = block[2],stride = 2)
        #self.layer4 = self.make_layer(inplaces = 1024,places = 512,block = block[3],stride = 2)

        #self.conv2 = Conv2(in_planes = 256,places = 512)
        #self.conv3 = Conv3(in_planes = 512,places = 512)

        self.conv4 = Conv4(in_planes = 512,places = 128)
        self.conv5 = Conv5(in_planes = 128,places = 2)

        #Eself.avgpool = nn.AvgPool2d(7,stride = 1)
        #self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode = 'fan_out',nonlinearity = 'relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def make_layer(self,inplaces,places,block,stride):
        layers = []
        layers.append(Bottleneck(inplaces,places,stride,downsampling = True))
        for i in range(1,block):
            layers.append(Bottleneck(places * self.expansion,places))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x) #1 64 256 320
        x = self.layer1(x) #1 256 256 320
        x = self.layer2(x) #1 512 128 320
        #x = self.layer3(x)
        #x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0),-1)
        #x = x.fc(x)
        #x = self.conv2(x) #1 512 256 320
        #x = self.conv3(x) #1 1024 128 160

        x = self.conv4(x) #1 128 60 76
        x = self.conv5(x) #1 2 60 76
        return x

def RestNet50():
    return RestNet([2,2,6,3])     





























