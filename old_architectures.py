'''
OLD

some architectures we experimented with.
'''
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # encoder
        self.enc_conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=4, padding=0)
        self.enc_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.enc_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.enc_maxpool2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.dec_unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0)
        self.dec_unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(8, 1, kernel_size=8, stride=4, padding=0)
        self.dec_conv4 = nn.ConvTranspose2d(1, 1, kernel_size=5, stride=1, padding=0)
        self.dec_conv5 = nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.dec_conv6 = nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.enc_conv1(x))
        x, i1 = self.enc_maxpool1(x)
        x = F.relu(self.enc_conv2(x))

        # x, i2 = self.enc_maxpool2(x)
        # x = self.dec_unpool2(x, i2)

        x = F.relu(self.dec_conv2(x))
        x = self.dec_unpool3(x, i1)
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv4(x))
        x = F.relu(self.dec_conv5(x))
        x = F.softmax(self.dec_conv6(x))
        return x


class R2_WNet(nn.Module):
    def __init__(self, n_channels, t=2, leaky=False, batch_norm=True, drop_rate=0.0,
                 dilate_first=1, dilate_second=1):
        super(R2_WNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)
        self.RRCNN1 = RRCNN_block(ch_in=n_channels, ch_out=64, t=t, leaky=leaky, batch_norm=batch_norm,
                                  drop_rate=drop_rate, dilation=dilate_first)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                                  dilation=dilate_first)
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                                  dilation=dilate_first)
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                                  dilation=dilate_first)
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                                  dilation=dilate_first)
        self.Up5 = up_conv(ch_in=1024, ch_out=512, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                           dilation=dilate_first)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t, leaky=leaky, batch_norm=batch_norm,
                                     drop_rate=drop_rate, dilation=dilate_first)
        self.Up4 = up_conv(ch_in=512, ch_out=256, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                           dilation=dilate_first)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t, leaky=leaky, batch_norm=batch_norm,
                                     drop_rate=drop_rate, dilation=dilate_first)
        self.Up3 = up_conv(ch_in=256, ch_out=128, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                           dilation=dilate_first)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t, leaky=leaky, batch_norm=batch_norm,
                                     drop_rate=drop_rate, dilation=dilate_first)
        self.Up2 = up_conv(ch_in=128, ch_out=64, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                           dilation=dilate_first)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                                     dilation=dilate_first)
        self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample2 = nn.Upsample(scale_factor=2)
        self.RRCNN12 = RRCNN_block(ch_in=65, ch_out=64, t=t, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                                   dilation=dilate_second)
        self.RRCNN22 = RRCNN_block(ch_in=64, ch_out=128, t=t, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                                   dilation=dilate_second)
        self.RRCNN32 = RRCNN_block(ch_in=128, ch_out=256, t=t, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                                   dilation=dilate_second)
        self.RRCNN42 = RRCNN_block(ch_in=256, ch_out=512, t=t, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                                   dilation=dilate_second)
        self.RRCNN52 = RRCNN_block(ch_in=512, ch_out=1024, t=t, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                                   dilation=dilate_second)
        self.Up52 = up_conv(ch_in=1024, ch_out=512, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                            dilation=dilate_second)
        self.Up_RRCNN52 = RRCNN_block(ch_in=1024, ch_out=512, t=t, leaky=leaky, batch_norm=batch_norm,
                                      drop_rate=drop_rate, dilation=dilate_second)
        self.Up42 = up_conv(ch_in=512, ch_out=256, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                            dilation=dilate_second)
        self.Up_RRCNN42 = RRCNN_block(ch_in=512, ch_out=256, t=t, leaky=leaky, batch_norm=batch_norm,
                                      drop_rate=drop_rate, dilation=dilate_second)
        self.Up32 = up_conv(ch_in=256, ch_out=128, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                            dilation=dilate_second)
        self.Up_RRCNN32 = RRCNN_block(ch_in=256, ch_out=128, t=t, leaky=leaky, batch_norm=batch_norm,
                                      drop_rate=drop_rate, dilation=dilate_second)
        self.Up22 = up_conv(ch_in=128, ch_out=64, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                            dilation=dilate_second)
        self.Up_RRCNN22 = RRCNN_block(ch_in=128, ch_out=64, t=t, leaky=leaky, batch_norm=batch_norm,
                                      drop_rate=drop_rate, dilation=dilate_second)
        self.Conv_1x12 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.RRCNN1(x)
        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        y1 = self.Up5(x5)
        y1 = torch.cat((x4, y1), dim=1)
        y1 = self.Up_RRCNN5(y1)
        y2 = self.Up4(y1)
        y2 = torch.cat((x3, y2), dim=1)
        y2 = self.Up_RRCNN4(y2)
        y3 = self.Up3(y2)
        y3 = torch.cat((x2, y3), dim=1)
        y3 = self.Up_RRCNN3(y3)
        y4 = self.Up2(y3)
        y4 = torch.cat((x1, y4), dim=1)
        y4 = self.Up_RRCNN2(y4)
        y5 = self.Conv_1x1(y4)
        z1 = self.RRCNN12(torch.cat((y5, y4), dim=1))
        z2 = self.Maxpool2(z1)
        z2 = self.RRCNN22(z2)
        z3 = self.Maxpool2(z2)
        z3 = self.RRCNN32(z3)
        z4 = self.Maxpool2(z3)
        z4 = self.RRCNN42(z4)
        z5 = self.Maxpool2(z4)
        z5 = self.RRCNN52(z5)
        w1 = self.Up52(z5)
        w1 = torch.cat((z4, w1), dim=1)
        w1 = self.Up_RRCNN52(w1)
        w2 = self.Up42(w1)
        w2 = torch.cat((z3, w2), dim=1)
        w2 = self.Up_RRCNN42(w2)
        w3 = self.Up32(w2)
        w3 = torch.cat((z2, w3), dim=1)
        w3 = self.Up_RRCNN32(w3)
        w4 = self.Up22(w3)
        w4 = torch.cat((z1, w4), dim=1)
        w4 = self.Up_RRCNN22(w4)
        w5 = self.Conv_1x12(w4)
        return torch.cat((y5, w5), dim=1)


class PWNet(nn.Module):
    def __init__(self, n_channels, model_name):
        super(PWNet, self).__init__()
        # Define prediction model
        runDir = "runs/"
        model_dir = runDir + model_name
        model_architecture = model_name.split("_")[1]
        self.premodel = R2U_Net(3, 1)
        self.premodel.load_state_dict(torch.load(model_dir + "/model/model.pt"))
        self.premodel.cuda()
        self.premodel.eval()

        # Define other layers
        self.inc = inconv(4, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 1)

    def forward(self, input):
        test = self.premodel(input)
        # TODO: Check whether concat is over the right dimension!
        con = torch.cat((input, test), 1)
        x1 = self.inc(con)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class LeakyUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LeakyUNet, self).__init__()
        self.inc = inconv(n_channels, 64, leaky=True)
        self.down1 = down(64, 128, leaky=True)
        self.down2 = down(128, 256, leaky=True)
        self.down3 = down(256, 512, leaky=True)
        self.down4 = down(512, 512, leaky=True)
        self.up1 = up(1024, 256, leaky=True)
        self.up2 = up(512, 128, leaky=True)
        self.up3 = up(256, 64, leaky=True)
        self.up4 = up(128, 64, leaky=True)
        self.outc = outconv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class LeakyR2Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(LeakyR2Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t, leaky=True)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t, leaky=True)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t, leaky=True)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t, leaky=True)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t, leaky=True)

        self.Up5 = up_conv(ch_in=1024, ch_out=512, leaky=True)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t, leaky=True)

        self.Up4 = up_conv(ch_in=512, ch_out=256, leaky=True)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t, leaky=True)

        self.Up3 = up_conv(ch_in=256, ch_out=128, leaky=True)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t, leaky=True)

        self.Up2 = up_conv(ch_in=128, ch_out=64, leaky=True)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t, leaky=True)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class PCNet(nn.Module):
    def __init__(self, n_channels, model_name):
        super(PCNet, self).__init__()
        # Define prediction model
        runDir = "runs/"
        model_dir = runDir + model_name
        model_architecture = model_name.split("_")[1]
        self.premodel = WNet(3, 16)
        self.premodel.load_state_dict(torch.load(model_dir + "/model/model.pt"))
        self.premodel.cuda()
        self.premodel.eval()

        self.conv1 = flexible_single_conv(in_ch=1, out_ch=16, size=101, padding=0, stride=1, dilation=1)
        self.conv2 = flexible_single_conv(in_ch=16, out_ch=16, size=101, padding=0, stride=1, dilation=1)
        self.conv3 = flexible_single_conv(in_ch=16, out_ch=32, size=101, padding=0, stride=1, dilation=1)
        self.outconv = outconv(in_ch=32, out_ch=1)

    def forward(self, input):
        pre_prediction = self.premodel(input)
        x1 = self.conv1(nn.functional.pad(pre_prediction, [50, 50, 50, 50], mode='reflect'))
        x2 = self.conv2(nn.functional.pad(x1, [50, 50, 50, 50], mode='reflect'))
        x3 = self.conv3(nn.functional.pad(x2, [50, 50, 50, 50], mode='reflect'))
        return F.sigmoid(self.outconv(x3))


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.xception_features = Xception(nInputChannels, os, pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.xception_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return F.sigmoid(x)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class LeakyR2NetNoSigmoid(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(LeakyR2NetNoSigmoid, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t, leaky=True)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t, leaky=True)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t, leaky=True)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t, leaky=True)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t, leaky=True)

        self.Up5 = up_conv(ch_in=1024, ch_out=512, leaky=True)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t, leaky=True)

        self.Up4 = up_conv(ch_in=512, ch_out=256, leaky=True)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t, leaky=True)

        self.Up3 = up_conv(ch_in=256, ch_out=128, leaky=True)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t, leaky=True)

        self.Up2 = up_conv(ch_in=128, ch_out=64, leaky=True)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t, leaky=True)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class UNet2(nn.Module):

    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1