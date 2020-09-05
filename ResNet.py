import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import os


class Basic_block(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, strides):
        super(Basic_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=strides, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shorcut = nn.Sequential()

        if strides != 1 or in_ch != self.expansion * out_ch:
            self.shorcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion))

    def forward(self, x):
        out = self.shorcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(x + out)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_class=200):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.pre = nn.Conv2d(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=1, padding=1)
        self.pre_bn = nn.BatchNorm2d(self.in_planes)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self.__make_layer(block, 64, strides=2, num_block=num_blocks[0])
        self.layer2 = self.__make_layer(block, 128, strides=1, num_block=num_blocks[1])
        self.layer3 = self.__make_layer(block, 256, strides=2, num_block=num_blocks[2])
        self.layer4 = self.__make_layer(block, 512, strides=1, num_block=num_blocks[3])

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048 * block.expansion, num_class)

    def __make_layer(self, block, out_ch, strides, num_block):
        layer = []
        strides = [strides] + [1] * (num_block - 1)
        for stride in strides:
            layer.append(block(self.in_planes, out_ch, stride))
            self.in_planes = out_ch * block.expansion

        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.pre(x)
        out = self.pre_bn(out)
        out = self.pool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # print(out.size(0))
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(Basic_block, [2, 2, 2, 2])


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class CGAN(nn.Module):
    """implements pix2pix model
    Generator: Unet Generator
    Discriminator: 3 Layer,
    Loss: BCE with Logits loss. (D not use sigmoid)
    """
    def __init__(self, input_nc, output_nc, num_downs, lambda_L1):
        super(CGAN, self).__init__()
        device = torch.device('cuda:0')
        self.netG = UnetGenerator(input_nc, output_nc, num_downs)
        self.netG = nn.DataParallel(self.netG)
        self.netG = self.netG.to(device)
        self.netD = NLayerDiscriminator(input_nc+output_nc)
        self.netD = nn.DataParallel(self.netD)
        self.netD = self.netD.to(device)

        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.lambda_L1 = lambda_L1

    def save(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'G_state_dict': self.netG.state_dict(),
            'D_state_dict': self.netD.state_dict(),
            'optD_state_dict': self.optimizer_D.state_dict(),
            'optG_state_dict': self.optimizer_G.state_dict(),
            }, os.path.join(path, 'GAN_{}.pt'.format(epoch)))

    def load(self, path):
        checkpoint = torch.load(path)
        self.netG.load_state_dict(checkpoint['G_state_dict'])
        self.netD.load_state_dict(checkpoint['D_state_dict'])

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, input, real):
        self.fake = self.netG(input)
        self.input = input
        self.real = real

        self.D_input_fake = torch.cat((self.input, self.fake), dim=1)
        self.D_input_real = torch.cat((self.input, self.real), dim=1)
        return self.fake, self.real

    def backward_D(self):
        pred_fake = self.netD(self.D_input_fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))

        pred_real = self.netD(self.D_input_real)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        self.loss_D = (loss_D_fake + loss_D_real)*0.5
        self.loss_D.backward()

    def backward_G(self):
        pred_fake = self.netD(self.D_input_fake.detach())
        loss_G_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = self.criterionL1(self.fake, self.real) * self.lambda_L1

        self.loss_G = loss_G_GAN + loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self, input, real):
        self.forward(input, real)
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        return self.loss_D, self.loss_G

    def get_output(self, input):
        self.set_requires_grad(self.netG, False)
        return self.netG(input)







