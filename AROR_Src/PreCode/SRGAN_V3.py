import torch
import torch.nn as nn
import pytorch_ssim

class ResBlock(nn.Module):
    def __init__(self, channals_out, channals_m, channals_in):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channals_in, channals_m, kernel_size=1, stride=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(channals_m, channals_m, kernel_size=1, stride=1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(channals_m, channals_out, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(0.1)

        self.cnn_skip = nn.Conv2d(channals_in, channals_out, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)

        if x.size(1) != residual.size(1):
            x = self.cnn_skip(x)
        x = x + residual
        x = self.relu3(x)
        return x

class ResBlock_up(nn.Module):
    def __init__(self, channals_out, channals_m, channals_in):
        super(ResBlock_up, self).__init__()
        self.conv1 = nn.Conv2d(channals_in, channals_m, kernel_size=1, stride=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(channals_m, channals_m, kernel_size=1, stride=1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(channals_m, channals_out, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(0.1)


    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)

        return residual

class down(nn.Module):
    def __init__(self, in_ch, m_ch, out_ch, kernel_size=4, padding=1, stride=2):
        super(down, self).__init__()
        self.same = nn.Sequential(
            ResBlock(channals_out=out_ch, channals_m=m_ch, channals_in=in_ch))
        self.d = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x_skip = self.same(x)
        x_down = self.d(x_skip)
        return x_skip, x_down

class up(nn.Module):
    def __init__(self, in_ch, m_ch, out_ch, kernel_size=4, padding=1, stride=2):
        super(up, self).__init__()
        self.u = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.1)
        )
        self.h = nn.Sequential(ResBlock_up(out_ch, m_ch, 2*in_ch))

    def forward(self, x, x_skip):
        x_up = self.u(x)
        x = torch.cat([x_up, x_skip], 1)
        x = self.h(x)
        return x


class down_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(down_conv, self).__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.1)
            )
    def forward(self, x):
        return self.cnn(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.s1_down = down(in_ch=1, m_ch=32, out_ch=64)#64x256x256
        self.s2_down = down(in_ch=64, m_ch=96, out_ch=128)#128x128x128
        self.s3_down = down(in_ch=128, m_ch=192, out_ch=256)#256x64x64
        self.s4_down = down(in_ch=256, m_ch=384, out_ch=512)#512x32x32
        self.down = down_conv(in_ch=512, out_ch=512, kernel_size=3, stride=1, padding=1)#512x16x16
        self.s4_up = up(in_ch=512, m_ch=640, out_ch=256)#256x32x32
        self.s3_up = up(in_ch=256, m_ch=320, out_ch=128)#128x64x64
        self.s2_up = up(in_ch=128, m_ch=160, out_ch=64)#64x128x128
        self.s1_up = up(in_ch=64, m_ch=80, out_ch=32)#32x256x256
        self.output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)#1x256x256

    def forward(self, x):
        (_,_,H,W) = x.size()
        x_s1_skip, x_s1_down = self.s1_down(x.view(-1, 1, H, W))
        x_s2_skip, x_s2_down = self.s2_down(x_s1_down)
        x_s3_skip, x_s3_down = self.s3_down(x_s2_down)
        x_s4_skip, x_s4_down = self.s4_down(x_s3_down)
        x_bottom = self.down(x_s4_down) 

        x_s4_up = self.s4_up(x_bottom, x_s4_skip)
        x_s3_up = self.s3_up(x_s4_up, x_s3_skip)
        x_s2_up = self.s2_up(x_s3_up, x_s2_skip)
        x_s1_up = self.s1_up(x_s2_up, x_s1_skip)
        out = self.output(x_s1_up)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1), #2D adaptive average pooling, output_size(batch_size,512,1,1) 
            nn.Conv2d(512, 1024, kernel_size=1), #dense layer
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1) #dense layer
        )

    def forward(self, x):
        batch_size = x.size(0)
        y = self.net(x).view(batch_size,-1) # flatten to (batch_size, 1)
        return y # Not the probability without sigmoid 
    

def G_initweights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            we = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            with torch.no_grad():
                 m.weight = nn.Parameter(0.1*we)  #Smaller initialization (x 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

def D_initweights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
#GAN Loss
def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function with sigmoid input.
    z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,), all 0 or all 1.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def MSE_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    - A PyTorch Tensor containing the MSE loss over the minibatch of input data.
    """
    MSE_fn = torch.nn.MSELoss(reduction='none')
    return MSE_fn(input, target).mean()
    
def SSIM_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    -  mean SSIM 
    """
    SSIM_fn = pytorch_ssim.SSIM(window_size = 11)
    return SSIM_fn(input, target)

def PCC_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    -  mean PCC
    """
    x_mean, y_mean = torch.mean(input, dim=[2,3], keepdim=True), torch.mean(target, dim=[2,3], keepdim=True)
    vx, vy = (input-x_mean), (target-y_mean)
    sigma_xy = torch.mean(vx*vy, dim=[2,3])
    sigma_x, sigma_y = torch.std(input, dim=[2,3]), torch.std(target, dim=[2,3])
    PCC = sigma_xy / ((sigma_x+1e-8) * (sigma_y+1e-8))
    return PCC.mean()

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    real_label = torch.ones_like(logits_real).to(logits_real.device)
    fake_label = torch.zeros_like(logits_fake).to(logits_fake.device)
    loss = bce_loss(logits_real, real_label) + bce_loss(logits_fake, fake_label)
    return loss

def generator_loss(logits_fake, real_data, fake_images):
    """
    Computes the generator loss: BCE+MSE+SSIM

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    - fake_images: PyTorch Tensor of shape (N,C,H,W)
    - real_data: PyTorch Tensor of shape (N,C,H,W)
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    lambda1, lambda2 = 5e-3, 2e-2
    label = torch.ones_like(logits_fake).to(logits_fake.device)
    bce = bce_loss(logits_fake, label) 
    mse = MSE_loss(fake_images, real_data) 
    ssim = -((1+SSIM_loss(fake_images, real_data))/2).log()
    return lambda1*bce + mse + lambda2*ssim

