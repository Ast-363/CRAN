import argparse


parser = argparse.ArgumentParser(description='BebyGAN')

# Hardware specifications
parser.add_argument('--gpu_id', type=str, help='specify GPU ID to use')
parser.add_argument('--num_workers', type=int, default=8)

# Train specificaions
parser.add_argument('--snap_path', type=str, default='./weights', help='path to save model weights')
parser.add_argument('--save_freq', type=str, default=10, help='save model frequency (epoch)')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/mnt/Dataset/anse_data/SRdata/DF2K', help='dataset root directory')
parser.add_argument('--patch_size', type=int, default=48, help='LR patch size') # default = 128 (in the paper)

# Train specifications
parser.add_argument('--epochs', type=int, default=5000, help='total epochs')
parser.add_argument('--batch_size', type=int, default=8, help='size of each batch') # default = 8 (in the paper)

# model specifications
parser.add_argument('--scale', type=int, default=4, help='super resolution scale')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of rgb')
parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')

# Optimizer specificaions
parser.add_argument('--lr_G', type=float, default=1e-4, help='initial learning rate of generator')
parser.add_argument('--lr_D', type=float, default=1e-4, help='initial learning rate of discriminator')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')

# Scheduler specifications
parser.add_argument('--step_size_G', type=int, default=2.0e5, help='step size of generator (iteration)')
parser.add_argument('--step_size_D', type=int, default=2.0e5, help='step size of discriminator (iteration)')
parser.add_argument('--gamma_G', type=float, default=0.5, help='generator learning rate decay ratio')
parser.add_argument('--gamma_D', type=float, default=0.5, help='discriminator learning rate decay ratio')

# checkpoint
parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')

# test specifications
parser.add_argument('--weights', type=str, help='load weights for test')
parser.add_argument('--dir_test', type=str, help='directory of test images')
parser.add_argument('--results', type=str, default='./results/', help='directory of test results')

args = parser.parse_args()