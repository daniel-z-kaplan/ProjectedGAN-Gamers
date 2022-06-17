# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.G_score = torch.tensor(500.0, requires_grad = False)
        self.D_score = torch.tensor(500.0, requires_grad = False)
        torch.autograd.set_detect_anomaly(True)
        self.k = 80 #Temp value
        
        
    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0
        def adjust_score(logits):
            
            k = self.k
            print("Received logits", logits)
            print('G_score', self.G_score)
            print('D_score', self.D_score)

            mean = torch.mean(logits).detach()
            print("Logits mean:", mean)
            change = torch.sub(torch.divide(torch.divide(self.G_score, self.D_score),2), mean) #So when scaling is .5 and mean is .6, discriminator is doing better than expected. Change = -.1, times K. Let's try this for now-ish..
            #.5 and .6, change = -.1, change * k = -2.4. G gains 2.4, D loses 2.4
            #.5 and .4, change is .1, change * k = 2.4. G loses 2.4, D gains 2.4
            #So we use change * k as the factor to change by
            #But it shouldn't keep going up when elo difference is high...?
            #We should use a higher BASE value, but actually have it stop going up as it goes up
            #It's a catchup mechanism
            print("Change factor:",change)
            print("Should change by this much:", torch.mul(change,k))
            self.G_score = torch.sub(self.G_score,torch.mul(change, k))
            self.D_score = torch.add(self.D_score,torch.mul(change, k))
            print('G_score', self.G_score)
            print('D_score', self.D_score)
            print("Adjusted scaling:",self.G_score/self.D_score)
            
        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()
                
                print("Maximize logits for generated")
                adjust_score(torch.sigmoid(gen_logits))
                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()
                
                #
                print("Minimize logits for generated via D")
                adjust_score(torch.sigmoid(gen_logits))#Send as is, because they wanna be zero.
                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                scaling = torch.div(self.G_score, self.D_score).to(self.device)
                loss_Dgen.mul(scaling).backward()#Changed now
#                 loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()
                
                
                print("Maximize logits for real via D")
                adjust_score(torch.sigmoid(1-real_logits))#Invert real logits, because they are meant to be 1
                
                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                scaling = torch.div(self.G_score, self.D_score).to(self.device)
                loss_Dreal.mul(scaling).backward()#Changed here
#                 loss_Dreal.backward()
