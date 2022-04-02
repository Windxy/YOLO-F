import torch
import torch.nn as nn
from collections import OrderedDict
from nets.CSPdarknet import darknet34

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

class SELayer(nn.Module):
    def __init__(self, channel, reduction=13):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 压缩空间
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()           # 1 x 13 x 224 x 224
        y = self.avg_pool(x)            # 1 x 13 x 1 x 1
        y = y.view(b, c)                # 1 x 13
        y = self.fc(y).view(b, c, 1, 1) # 1 x 13 x 1 x 1
        y = y.expand_as(x)              # 1 x 13 x 224 x 224
        # print(x.shape,y.shape)
        return x * y


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        self.atten = True
         # backbone
        self.backbone = darknet34(None)

        # self.conv1 = make_three_conv([512,1024],1024)       #x x 512
        # self.SPP = SpatialPyramidPooling()
        # self.conv2 = make_three_conv([512,1024],2048)

        # self.conv_for_P4 = conv2d(512,256,1)
        # self.make_five_conv1 = make_five_conv([256, 512],512)
        #
        # self.conv_for_P3 = conv2d(256,128,1)
        # self.make_five_conv2 = make_five_conv([128, 256],256)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        # final_out_filter2 = num_anchors * (5 + num_classes)
        #
        # self.down_sample1 = conv2d(128,256,3,stride=2)
        # self.make_five_conv3 = make_five_conv([256, 512],512)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # final_out_filter1 =  num_anchors * (5 + num_classes)
        #
        #
        # self.down_sample2 = conv2d(256,512,3,stride=2)
        # self.make_five_conv4 = make_five_conv([512, 1024],1024)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter =  num_anchors * (5 + num_classes)

        self.yolo_head3 = yolo_head([256, final_out_filter],512)
        self.yolo_head2 = yolo_head([512, final_out_filter],256)
        self.yolo_head1 = yolo_head([1024, final_out_filter],512)

        # atte
        self.atten0 = SELayer(1024)
        self.atten1 = SELayer(512)
        self.atten2 = SELayer(512)
        self.atten3 = SELayer(256)
        self.atten4 = SELayer(256)

        self.conv0 = BasicConv(in_channels=1024,out_channels=512,kernel_size=3,stride=1)
        self.upsample0 = Upsample(512,512)

        self.conv1 = BasicConv(1024,512,3,2)
        self.upsample1 = Upsample(512,512)

        self.conv2 = BasicConv(1024,256,3,1)
        self.upsample2 = Upsample(256,256)

        self.conv3 = BasicConv(256*2,256,3,2)
        self.upsample3 = Upsample(256,256)







    def forward(self, x):
        #  backbone
        # x2, x1, x0 = self.backbone(x)
        '''        '''
        '''
        x4:256 52 52
        x3:256 52 52
        x2:512 26 26
        x1:512 26 26
        x0:1024 13 13
        '''
        x4,x3,x2,x1,x0 = self.backbone(x)

        if self.atten:
            x4 = self.atten4(x4)
            x3 = self.atten3(x3)
            x2 = self.atten2(x2)
            x1 = self.atten1(x1)
            x0 = self.atten0(x0)


        # P5 = self.conv1(x0)
        # P5 = self.SPP(P5)        # TODO：验证没有SPP的情况如何
        # P5 = self.conv2(P5)

        x0 = self.conv0(x0)
        x0 = self.upsample0(x0)

        x1 = torch.cat([x0,x1],axis=1)
        x1 = self.conv1(x1)
        out1 = self.yolo_head1(x1)
        x1 = self.upsample1(x1)

        x2 = torch.cat([x1,x2],axis=1)
        x2 = self.conv2(x2)
        x2 = self.upsample2(x2)

        x3 = torch.cat([x2,x3],axis=1)
        x3 = self.conv3(x3)
        out2 = self.yolo_head2(x3)
        x3 = self.upsample3(x3)

        x4 = torch.cat([x3,x4],axis=1)
        out3 = self.yolo_head3(x4)


        return out1, out2, out3

def model_structure(model):
        blank = ' '
        print('-' * 90)
        print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
              + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
              + ' ' * 3 + 'number' + ' ' * 3 + '|')
        print('-' * 90)
        num_para = 0
        type_size = 1  ##如果是浮点数就是4

        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (30 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (40 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            print('| {} | {} | {} |'.format(key, shape, str_num))
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para))
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)

if __name__ == '__main__':

    model = YoloBody(3,1)
    print()
    x = torch.rand(1,3,416,416)
    y = model(x)
    for i in range(len(y)):
        print(y[i].size())
    model_structure(model)

