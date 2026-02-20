import torch
import torch.nn as nn
import torch.optim as optim

from model_pix2pix import Discriminator, Generator
from torch.utils.data import DataLoader
from dataset import PopJazzDataset
import config
from tqdm import tqdm

import utils
def train_fn(disc_p, disc_j, gen_p, gen_j, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler):
    # This is for visualizing, it creates the progress bar, which is cool
    loop = tqdm(loader, leave = True)
    
    for idx, (pop, jazz) in enumerate(loop):
        pop = pop.to(config.DEVICE)
        jazz = jazz.to(config.DEVICE)
        # Training loop
        ######################FOR THE DISCRIMINATOR#################
        # Pipeline : Get some fake stuff + Real stuff -> Plug in the D -> Construct the loss -> Backprop
        with torch.amp.autocast(device_type= config.DEVICE): # Put things in the autocast so that these computations are perform with float16
            # Pop
            fake_pop = gen_p(jazz)
            disc_p_real = disc_p(pop)
            disc_p_fake = disc_p(fake_pop.detach())  # Detach this fake_pop because we'll use it later
            loss_disc_p_real = MSE(disc_p_real, torch.ones_like(disc_p_real))
            loss_disc_p_fake = MSE(disc_p_fake, torch.zeros_like(disc_p_fake))
            loss_disc_p = loss_disc_p_real + loss_disc_p_fake
            
            # Jazz 
            fake_jazz = gen_j(pop)
            disc_j_real = disc_j(jazz)
            disc_j_fake = disc_j(fake_jazz.detach()) 
            loss_disc_j_real = MSE(disc_j_real, torch.ones_like(disc_j_real))
            loss_disc_j_fake = MSE(disc_j_fake, torch.zeros_like(disc_j_fake))
            loss_disc_j = loss_disc_j_real + loss_disc_j_fake
            
            # Put things together
            loss_disc = (loss_disc_j + loss_disc_p) / 2
        # Put back to float32 to calculate the grad
        opt_disc.zero_grad()
        d_scaler.scale(loss_disc).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        #################FOR THE GENERATOR#####################
        # Normal pipeline: Put the fake images in the discriminator -> Set the label = 1 -> Backprop
        # What's new: Add up 3 Losses
        # 1. Adversarial Loss: Normal stuff (MSE)
        # 2. Cycle Loss: jazz --gen_p--> fake_pop --gen_j--> jazz (L1 loss)
        # 3. Identity Loss: jazz --gen_j--> jazz (L1 loss)
        with torch.amp.autocast(device_type= config.DEVICE):
            disc_p_fake_g = disc_p(fake_pop)
            disc_j_fake_g = disc_j(fake_jazz)
            
            # Adversarial Loss
            # Set the label to one for the generator to fool the disc
            ad_loss_p = MSE(disc_p_fake_g, torch.ones_like(disc_p_fake_g))
            ad_loss_j = MSE(disc_j_fake_g, torch.ones_like(disc_j_fake_g))
            
            # Cycle loss
            out_p = gen_p(fake_jazz)
            out_j = gen_j(fake_pop)
            cyc_loss_p = L1(out_p, pop)
            cyc_loss_j = L1(out_j, jazz)
            
            # Identity loss
            id_loss_p = L1(pop, gen_p(pop))
            id_loss_j = L1(jazz, gen_j(jazz))
            
            # Add up
            loss_g = ad_loss_j + ad_loss_p + config.LAMBDA_CYC * (cyc_loss_j + cyc_loss_p) + config.LAMBDA_ID * (id_loss_j + id_loss_p)
            
        opt_gen.zero_grad()
        g_scaler.scale(loss_g).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        # Update progress bar
        loop.set_postfix(loss_g=loss_g.item(), loss_d=loss_disc.item())
        
def main():
    # 1. Lock in the random seed for reproducible results!
    utils.seed_everything(42)
    # Initializing 2 discriminators and 2 generators (pop and jazz)
    disc_p = Discriminator().to(config.DEVICE)
    gen_p = Generator().to(config.DEVICE)

    disc_j = Discriminator().to(config.DEVICE)
    gen_j = Generator().to(config.DEVICE)
    
    # Initialize the Adam solver, with the betas specified in the original paper
    opt_disc = optim.Adam(list(disc_p.parameters()) + list(disc_j.parameters()), lr = config.LR, betas = (0.5, 0.999))
    opt_gen = optim.Adam(list(gen_p.parameters()) + list(gen_j.parameters()), lr = config.LR, betas = (0.5, 0.999))
    
    # 2. Load Checkpoints (if config is set to load previous weights)
    if config.LOAD_MODEL:
        utils.load_checkpoint("disc_p.pth.tar", disc_p, opt_disc, config.LR)
        utils.load_checkpoint("disc_j.pth.tar", disc_j, opt_disc, config.LR)
        utils.load_checkpoint("gen_p.pth.tar", gen_p, opt_gen, config.LR)
        utils.load_checkpoint("gen_j.pth.tar", gen_j, opt_gen, config.LR)
    
    # Data
    dataset = PopJazzDataset('pop_train', 'jazz_train')
    loader = DataLoader(dataset, batch_size= config.BATCH_SIZE, shuffle = True)
    
    # Specify the losses, in this case we need to use the L1 loss and MSE, not the BCE anymore
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    
    # Playing around with float16 for speed boosting and memory saving (learned this from the Aladdin guy)
    d_scaler = torch.amp.GradScaler()
    g_scaler = torch.amp.GradScaler()
    
    for epoch in range(config.EPOCHS):
        print(f"Epoch: [{epoch}/{config.EPOCHS}]")
        train_fn(disc_p, disc_j, gen_p, gen_j, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler)
        
        # 3. Save model states and optimizers so you don't lose progress
        if config.SAVE_MODEL:
            utils.save_checkpoint(disc_p, opt_disc, filename="disc_p.pth.tar")
            utils.save_checkpoint(disc_j, opt_disc, filename="disc_j.pth.tar")
            utils.save_checkpoint(gen_p, opt_gen, filename="gen_p.pth.tar")
            utils.save_checkpoint(gen_j, opt_gen, filename="gen_j.pth.tar")
            
        # 4. Grab a batch of data and save the visual spectrograms 
        pop_sample, jazz_sample = next(iter(loader))
        pop_sample, jazz_sample = pop_sample.to(config.DEVICE), jazz_sample.to(config.DEVICE)
        utils.save_spectrogram_samples(gen_j, gen_p, pop_sample, jazz_sample, epoch)
if __name__ == "__main__":
    main()
