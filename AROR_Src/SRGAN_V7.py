import torch
import torch.nn as nn
import pytorch_ssim
from torch.autograd import Variable
import numpy as np


# Adjustments (July 20)
# (1) GroupNorm: Group=32 ; (2) MAE & FMAE in cuda, fine-tine lambda1;
# (3) Modify ResBlock & downsampling/unsampling -> Conv/ConvTrans+GroupNorm+LeakReLU

class ResBlock(nn.Module):
    def __init__(self, channals_out, channals_m, channals_in, Res_bool):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channals_in, channals_m, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channals_m, channals_m, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channals_m, channals_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
            )
        
        self.ResBool = Res_bool
        self.cnn_skip = nn.Conv2d(channals_in, channals_out, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        if self.ResBool:
            if x.size(1) != residual.size(1):
                x = self.cnn_skip(x)
            Fx = x + residual
        else:
            Fx = residual 
        return Fx


class down(nn.Module):
    def __init__(self, in_ch, m_ch, out_ch, kernel_size=4, padding=1, stride=2):
        super(down, self).__init__()
        self.conv_down = nn.Sequential(
            ResBlock(channals_out=out_ch, channals_m=m_ch, channals_in=in_ch, Res_bool=True))
        self.d = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.GroupNorm(32, out_ch),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = self.conv_down(x)
        x_skip, x_down = x, self.d(x)        
        return x_skip, x_down

class up(nn.Module):
    def __init__(self, in_ch, m_ch, out_ch, kernel_size=4, padding=1, stride=2):
        super(up, self).__init__()
        self.u = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.GroupNorm(32, in_ch),
            nn.LeakyReLU(0.1)
        )
        self.conv_up = nn.Sequential(ResBlock(out_ch, m_ch, 2*in_ch, Res_bool=False))

    def forward(self, x, x_skip): 
        x_up = self.u(x)
        Fx = torch.cat([x_up, x_skip], 1)
        Fx = self.conv_up(Fx)
        return Fx


class down_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(down_conv, self).__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(32, out_ch),
            nn.LeakyReLU(0.1)
            )
    def forward(self, x):
        return self.cnn(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.s1_down = down(in_ch=1, m_ch=32, out_ch=64)
        self.s2_down = down(in_ch=64, m_ch=96, out_ch=128)
        self.s3_down = down(in_ch=128, m_ch=192, out_ch=256)
        self.s4_down = down(in_ch=256, m_ch=384, out_ch=512)
        self.down = down_conv(in_ch=512, out_ch=512, kernel_size=3, stride=1, padding=1)
        self.s4_up = up(in_ch=512, m_ch=640, out_ch=256)
        self.s3_up = up(in_ch=256, m_ch=320, out_ch=128)
        self.s2_up = up(in_ch=128, m_ch=160, out_ch=64)
        self.s1_up = up(in_ch=64, m_ch=80, out_ch=32)
        self.output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

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
        if x.size(1) != 1:
            x = nn.Conv2d(x.size(1), 1, kernel_size=3, padding=1).cuda()(x)

        y = self.net(x) # flatten to (batch_size, 1)
        return torch.sigmoid(y.view(x.size(0),-1)) #the probability 
    

def G_initweights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            we = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            with torch.no_grad():
                 m.weight = nn.Parameter(0.1*we)  #Smaller initialization (x 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d,  nn.GroupNorm, nn.LayerNorm)):  #nn.InstanceNorm2d,
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

def D_initweights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(0.1*m.weight, mode='fan_out', nonlinearity='leaky_relu')            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):  #nn.InstanceNorm2d,
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

#GAN Loss
datype = torch.cuda.FloatTensor

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

def SquCE_loss(input, target):
    """
    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.   (x)
    - target: PyTorch Tensor of shape (N,), all 0 or all 1. (z)

    Returns:
    - The mean square of classification error over the minibatch of input data.
    z * (1-x)^2 +(1-z) * x^2
    """
    SquCE = target * (1-input)**2 + (1-target) * input**2
    return SquCE.mean()


def MAE_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    - A PyTorch Tensor(N,) containing the MSE loss over the minibatch of input data.
    """
    MAE_fn = torch.nn.L1Loss(reduction='mean')
    return MAE_fn(input, target)

# def FMSE_loss(input, target):  #ndarray->tensor
#     """
#      Inputs:
#     - input: PyTorch Tensor of shape (N, C, H, W) .
#     - target: PyTorch Tensor of shape (N, C, H, W).

#     Returns:
#     - A PyTorch Tensor containing the MSE loss over the minibatch of input data.
#     """
#     MSE_fn = torch.nn.MSELoss(reduction='none')
#     Finput, Ftarget = np.real(np.fft.fft2(input.data.cpu().numpy())), np.real(np.fft.fft2(target.data.cpu().numpy()))
#     Finput, Ftarget = torch.tensor(Finput, requires_grad=True).type(datype), torch.tensor(Ftarget, requires_grad=True).type(datype)
#     return MSE_fn(Finput, Ftarget).mean()


def FMAE_loss(input, target): 
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    - MAE loss over the minibatch of the DFT of input data.

    """
    MAE_fn = torch.nn.L1Loss(reduction='mean')
    [N,C,H,W] = input.size()
    AmpFinput, AmpFtarget = torch.zeros(N,H,W).type(datype), torch.zeros(N,H,W).type(datype)
    for i in range(N):
        Finput, Ftarget = torch.rfft(input[i,0,:,:],2,False,False), \
                            torch.rfft(target[i,0,:,:],2,False,False)
        AmpFinput[i,:,:] = torch.sqrt(Finput[:,:,0]**2+Finput[:,:,1]**2)
        AmpFtarget[i,:,:] = torch.sqrt(Ftarget[:,:,0]**2+Ftarget[:,:,1]**2)

    MinFinput = torch.min(torch.min(AmpFinput,dim=1)[0],dim=1)[0].reshape(N,C,1,1)
    MaxFinput = torch.max(torch.max(AmpFinput,dim=1)[0],dim=1)[0].reshape(N,C,1,1)
    MinFtarget = torch.min(torch.min(AmpFtarget,dim=1)[0],dim=1)[0].reshape(N,C,1,1)
    MaxFtarget = torch.max(torch.max(AmpFtarget,dim=1)[0],dim=1)[0].reshape(N,C,1,1)
    NormFinput = (AmpFinput-MinFinput) / (MaxFinput-MinFinput)
    NormFtarget = (AmpFtarget-MinFtarget) / (MaxFtarget-MinFtarget)
    Ferror = MAE_fn(NormFinput, NormFtarget)  #Better to normalize before L1 loss
    return Ferror


def SSIM_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    -  mean SSIM 
    """
    SSIM_fn = pytorch_ssim.SSIM(window_size = 11)
    ssim = SSIM_fn(input, target)
    return ssim
    
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

def TV_loss(input):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .

    Returns:
    - total variation of the input
    """
    [N,C,H,W] = input.size()
    TempBatch = torch.zeros([N,C,H+2,W+2]).type(datype)
    TV = 0
    for i in range(0,N):
        TempBatch[i,0,1:H+1,1:W+1] = input[i,0,:,:].clone()
        Temp1, Temp2 = torch.squeeze(TempBatch[i,:,2:H+2,1:W+1]), torch.squeeze(TempBatch[i,:,1:H+1,2:W+2])
        TV = TV + 1/(H*W) * ((Temp1[0:H-1,:]-input[i,0,0:H-1,:]).pow(2).sum() + (Temp2[:,0:W-1]-input[i,0,:,0:W-1]).pow(2).sum())
    return TV/N


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
    # loss = SquCE_loss(logits_real, real_label) + SquCE_loss(logits_fake, fake_label)
    loss = bce_loss(logits_real, real_label) + bce_loss(logits_fake, fake_label)
    return loss

def generator_loss(logits_fake, real_data, fake_images,lambd):
    """
    Computes the generator loss: BCE+MSE+SSIM

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    - fake_images: PyTorch Tensor of shape (N,C,H,W)
    - real_data: PyTorch Tensor of shape (N,C,H,W)
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    label = torch.ones_like(logits_fake).to(logits_fake.device)
    SquCE = SquCE_loss(logits_fake, label) 
    # bce = bce_loss(logits_fake, label) 
    mae = MAE_loss(fake_images, real_data)
    fmae = FMAE_loss(fake_images, real_data)
    # lambd = (mae+0.01*fmae)/(4*SquCE)
    lambd = 1e-3
    return (mae+0.01*fmae+lambd*SquCE, mae+0.01*fmae, SquCE)


def clip_grad_value_(parameters, clip_value):
    """Clips gradient of an iterable of parameters at specified value.
    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)

# def test_grad_value(parameters, optimizer):
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     Nonan = True
#     for p in parameters:
#         if p.grad is not None:
#             if torch.isnan(p.grad).any():
#                 Nonan = False
#                 break  #终止最小封闭for循环
#     if Nonan:
#         optimizer.step()

def test_grad_value(parameters, optimizer):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    GradIter = filter(lambda p: p.grad is not None, parameters)
    GradList = list(filter(lambda p: torch.isnan(p.grad).any(), GradIter))
    if not GradList:
        optimizer.step()