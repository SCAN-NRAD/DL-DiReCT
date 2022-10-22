import numpy as np
import sys
import os
import argparse
import csv
import pandas as pd
import nibabel as nib
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, ReplicationPad2d, Dropout
from torch.utils.data.sampler import Sampler, SequentialSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from numpy import require


# Save logits of each class
# Set to None to save all:
#    SAVE_LOGITS_FILTER = None
# Set to empty list to disable save of logits:
#    SAVE_LOGITS_FILTER = []
# Set to list of selected labels otherwise (e.g. for DL+DiReCT):
#    SAVE_LOGITS_FILTER = ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter', 'WM-hypointensities',
#                          'Left-Cerebral-Cortex', 'Right-Cerebral-Cortex',
#                          'Left-Amygdala', 'Right-Amygdala', 'Left-Hippocampus', 'Right-Hippocampus']
SAVE_LOGITS_FILTER = ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter', 'WM-hypointensities',
                      'Left-Cerebral-Cortex', 'Right-Cerebral-Cortex',
                      'Left-Amygdala', 'Right-Amygdala', 'Left-Hippocampus', 'Right-Hippocampus']


stack_depth = 7
BATCH_SIZE = 10
VERBOSE = True
DIM = 216
SCRIPT_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))


def get_label_def(label_names):
    # To generate the hard segmentations, we use the FS labels where we have a 1:1 correspondence,
    # otherwise a label above 100 (not used in FS)
    LUT = dict()
    with open('{}/fs_lut.csv'.format(SCRIPT_DIR), 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            LUT[row[1]] = int(row[0])
            
    LABELS = dict()
    for idx, label in enumerate(label_names):
        if label in LUT:
            l = LUT[label]
        else:
            l = idx + 100
        
        LABELS[label] = l
        
    return LABELS
    

def get_stack(axis, volume, central_slice, first_slice=None, last_slice=None,
              stack_depth=5, size = (DIM,DIM), lower_threshold = None, upper_threshold= None, return_nonzero= True):

    image_data = np.array(np.swapaxes(volume, 0, axis), copy = True)
    mean = np.mean(image_data[image_data>0])
    sd = np.sqrt(np.var(image_data[image_data>0]))
    
    if lower_threshold is None:
        lower_threshold = 0
    
    if upper_threshold is None:
        upper_threshold = np.percentile(image_data[image_data>0], 99.9)
    
    image_data[image_data<lower_threshold] = lower_threshold
    image_data[image_data>upper_threshold] = upper_threshold
    
    if first_slice is None:
        if central_slice is not None:
            first_slice = central_slice - stack_depth//2
            last_slice = central_slice + stack_depth//2 + 1
        elif last_slice is not None:
            first_slice = last_slice - stack_depth
    elif last_slice is None:
            last_slice = min(first_slice + stack_depth, len(image_data))
    pad_up = max(0, -first_slice)

    pad_down = -min(0, len(image_data)-last_slice)

    first_slice = max(first_slice,0)
    last_slice = min(last_slice, len(image_data))
    initial_stack = image_data[first_slice:last_slice]
    initial_shape = initial_stack.shape[1:]
    shape_difference = (size[0] - initial_shape[0],size[1] - initial_shape[1])
    pad_size = ((pad_up,pad_down),
                (max(0, shape_difference[0]//2), max(0, shape_difference[0] - shape_difference[0]//2)),
                (max(0, shape_difference[1]//2), max(0, shape_difference[1] - shape_difference[1]//2))
               )
    initial_stack = np.pad(initial_stack, pad_size, mode = 'constant', constant_values = lower_threshold)
    
    if return_nonzero :
        nonzero_mask = (initial_stack>lower_threshold).astype(np.uint8)
    else:
        nonzero_mask = None
    
    return (initial_stack - mean)/sd, nonzero_mask


def get_stack_no_augment(axis, volume, first_slice, last_slice, size=(DIM, DIM)):
    return get_stack(axis = axis, volume = volume, central_slice=None, stack_depth=None, first_slice = first_slice, last_slice = last_slice, size=size)


def reduce_3d_depth (in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad3d((1,1,1,1,0,0))),
            ("conv1", nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm3d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            #("dropout", nn.Dropout(p=0.2))
    ]))
    return layer

def down_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

def up_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

class DilatedDenseUnit(nn.Module):
    def __init__(self, in_channel, growth_rate , kernel_size, dilation):
        super(DilatedDenseUnit,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", nn.InstanceNorm2d(in_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("pad1", nn.ReplicationPad2d(dilation)),
            ("conv1", nn.Conv2d(in_channel, growth_rate, kernel_size=kernel_size, dilation = dilation,padding=0)),
            ("dropout", nn.Dropout(p=0.0))]))
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = concatenate(x, out)
        return out
    
class AttentionModule(nn.Module):
    def __init__(self, in_channel , intermediate_channel, out_channel, kernel_size=3):
        super(AttentionModule,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", nn.InstanceNorm2d(in_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, intermediate_channel, kernel_size=kernel_size,padding=0)),
            ("bn2", nn.InstanceNorm2d(intermediate_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(intermediate_channel, out_channel, kernel_size=kernel_size,padding=0)),
            ("sigmoid", nn.Sigmoid())]))
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = x * out
        return out
    

def center_crop(layer, target_size):
    _, _, layer_width, layer_height = layer.size()
    start = (layer_width - target_size) // 2
    crop = layer[:, :, start:(start + target_size), start:(start + target_size)]
    return crop

def concatenate(link, layer):
    concat = torch.cat([link, layer], 1)
    return concat

def dense_atrous_bottleneck(in_channel, growth_rate = 12, depth = [4,4,4,4]):
    layer_dict = OrderedDict()
    for idx, growth_steps in enumerate(depth):
        dilation_rate = 2**idx
        for y in range(growth_steps):
            layer_dict["dilated_{}_{}".format(dilation_rate,y)] = DilatedDenseUnit(in_channel, 
                                                                        growth_rate, 
                                                                        kernel_size=3, 
                                                                        dilation = dilation_rate)
            in_channel = in_channel + growth_rate
        
        layer_dict["attention_{}".format(dilation_rate)] = AttentionModule(in_channel, in_channel//4, in_channel)
        
    return nn.Sequential(layer_dict), in_channel


class UNET_3D_to_2D(nn.Module):
    def __init__(self, depth, channels_in = 1, 
                 channels_2d_to_3d = 32, channels = 32, output_channels = 1, slices=stack_depth, 
                 dilated_layers = [4,4,4,4],
                growth_rate = 12):
        super(UNET_3D_to_2D, self).__init__()
        self.main_modules = []
        
        self.depth = depth
        self.slices = slices
        
        self.depth_reducing_layers = ModuleList([reduce_3d_depth(in_channel, channels_2d_to_3d, kernel_size=3, padding=0)
                                                 for in_channel in [channels_in]+[channels_2d_to_3d]*(slices//2 - 1)])
        
        
        self.down1 = down_layer(in_channel=channels_2d_to_3d, out_channel=channels, kernel_size=3, padding=0)
        self.main_modules.append(self.down1)
        self.max1 = nn.MaxPool2d(2)
        self.down_layers = ModuleList([down_layer(in_channel = channels*(2**i), 
                                  out_channel = channels * (2**(i+1)),
                                  kernel_size = 3,
                                  padding=0
                                 ) for i in range(self.depth)])
        self.main_modules.append(self.down_layers)
        self.max_layers = ModuleList([nn.MaxPool2d(2) for i in range(self.depth)])
        
        self.bottleneck, bottleneck_features  = dense_atrous_bottleneck(channels*2**self.depth, growth_rate = growth_rate, 
                                                                       depth = dilated_layers)
        self.main_modules.append(self.bottleneck)
        
        self.upsampling_layers = ModuleList([nn.Sequential(OrderedDict([
                ("upsampling",nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)),
                ("pad", nn.ReplicationPad2d(1)),
                ("conv", nn.Conv2d(in_channels= bottleneck_features, 
                                   out_channels=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0))]))  for i in range(self.depth, -1, -1)])
        self.main_modules.append(self.upsampling_layers)
        self.up_layers = ModuleList([up_layer(in_channel= bottleneck_features+ channels*(2**(i)), 
                                   out_channel=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0) for i in range(self.depth, -1, -1)])
        
        self.main_modules.append(self.up_layers)
        self.last = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)
        self.main_modules.append(self.last)
        
        self.logvar = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        # down
        out = x
        
        for i in range(self.slices//2):
            out = self.depth_reducing_layers[i](out)
        
        out.transpose_(1, 2).contiguous()
        size = out.size()
        out = out.view((-1, size[2], size[3], size[4]))
        
        links = []
        out = self.down1(out)
        links.append(out)
        out = self.max1(out)
        
        for i in range(self.depth):
            out = self.down_layers[i](out)
            links.append(out)
            out = self.max_layers[i](out)
        
        out = self.bottleneck(out)
        
        links.reverse()

        # up
        for i in range(self.depth+1):

            out = self.upsampling_layers[i](out)

            out = concatenate(links[i], out)
            out = self.up_layers[i](out)

        pred = self.last(out)
        logvar = self.logvar(out)
        logvar = -torch.exp(logvar)

        return pred, logvar

    

def apply_to_case(model, volumes, batch_size, stack_depth = stack_depth, axes=[0], size=(DIM,DIM), mask_bg = True, lowmem=False):
    volume_0 = volumes[0]
    ensemble_logits = []

    for axis in axes:
        print('Axis {}'.format(axis), end='', flush=True) if VERBOSE else False
        logit_total = []
    
        num_batches = volume_0.shape[axis]//(batch_size)
        if volume_0.shape[axis]%batch_size > 0:
            num_batches = num_batches + 1

        padding = stack_depth//2
        
        class BrainDataTest(Dataset):
            def __init__(self):
                self.length = num_batches

            def __getitem__(self, batch):
                first_slice = batch*batch_size - padding
                last_slice = np.min([(batch+1)*batch_size+padding, volume_0.shape[axis]+padding])
                extra_upper_slices = np.max([0, stack_depth - (last_slice - first_slice)])
                last_slice = last_slice + extra_upper_slices

                images_t1, nonzero_masks = get_stack_no_augment(axis = axis, 
                                                                volume = volume_0, 
                                                                first_slice=first_slice, 
                                                                last_slice=last_slice,
                                                               size=size)

                images = np.stack([images_t1])

                if padding >0:
                    nonzero_masks = nonzero_masks[padding:-(padding+extra_upper_slices)]

                return images.astype(np.float32), nonzero_masks.astype(np.float32)   

            def __len__(self):
                return self.length    

        test_generator = DataLoader(BrainDataTest(), sampler = SequentialSampler(BrainDataTest()), 
                         num_workers=0,pin_memory=True)
        for images, nonzero_masks  in test_generator:
            print('.', end='', flush=True) if VERBOSE else False
            images = images.to(device)
            nonzero_mask = nonzero_masks.to(device)

            outputs, logit_flip = model(images)

            if mask_bg:
                outputs = outputs * torch.unsqueeze(nonzero_mask[0],1)

            out_cpu = outputs.cpu().data.numpy()
            if lowmem:
                out_cpu = out_cpu.astype(np.float16)
                
            logit_total.append(out_cpu)


        print('') if VERBOSE else False
        full_logit = np.concatenate(logit_total)
        new_shape = full_logit[:, 0, :, :].shape

        shape_difference = (new_shape[0] - np.swapaxes(volume_0,0, axis).shape[0],
                            new_shape[1]-np.swapaxes(volume_0,0, axis).shape[1],
                            new_shape[2]-np.swapaxes(volume_0,0, axis).shape[2])

        full_logit = np.swapaxes(full_logit, 1, 0)
        full_logit = full_logit[:, shape_difference[0]//2:new_shape[0]- (shape_difference[0] - shape_difference[0]//2),
                               shape_difference[1]//2: new_shape[1]- (shape_difference[1] - shape_difference[1]//2),
                               shape_difference[2]//2: new_shape[2]- (shape_difference[2] - shape_difference[2]//2)]
        full_logit = np.swapaxes(full_logit, 1, axis+1)

        ensemble_logits.append(full_logit)
        
    
    return np.mean(np.array(ensemble_logits),axis=0)


def locate_model(model_file):
    if os.path.exists(model_file):
        return model_file
    
    for prefix in ['', '{}/../model/'.format(SCRIPT_DIR)]:
        for suffix in ['.pth', '_f1.pth']:
            file = '{}{}{}'.format(prefix, model_file, suffix)
            if os.path.exists(file):
                return file
    
        
def load_checkpoint(checkpoint_file, device):
    if not os.path.exists(checkpoint_file):
        print('Error: model {} not found'.format(checkpoint_file))
        sys.exit(1)
        
    print('loading checkpoint {}'.format(checkpoint_file)) if VERBOSE else False
    return torch.load(checkpoint_file, map_location=device)


def validate_input(t1, t1_data):
    # input sanity check: we expect images in LIA orientation (FS space)!
    orientation = ''.join(nib.aff2axcodes(t1.affine))
    if orientation != 'LIA':
        print('\nWARNING: Invalid orientation found for {}: {}\n'.format(subject_id, orientation))
        
    # check for non-zero corners (background)
    corner_idx = np.ix_((0,-1),(0,-1),(0,-1))
    if not np.allclose(t1_data[corner_idx].flatten(), 0):
        print('\nWARNING: Non-zero voxels detected in background (corners). Make sure input is brain extracted (use --bet) and background intensities are exactly 0\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSCAN: Deep learning based anatomy segmentation and cortex parcellation')
    parser.add_argument("--model", required=False, default='v0_f1')
    parser.add_argument("--lowmem", required=False, default=False)
    parser.add_argument("T1w")
    parser.add_argument("destination_dir")
    parser.add_argument("subject_id")
    
    args = parser.parse_args()
    t1_file = args.T1w
    output_dir = args.destination_dir
    model_file = locate_model(args.model)
    subject_id = args.subject_id
    
    if not os.path.exists(t1_file):
        print('T1w file {} not found'.format(t1_file))
        sys.exit(1)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = load_checkpoint(model_file, device)
    target_label_names = checkpoint['label_names']
    # number of last labels to ignore for hard segmentation (argmax), e.g. left-hemi, right-hemi, brain
    NUM_IGNORE_LABELS = checkpoint['label_num_ignore']
    LABELS = get_label_def(target_label_names)
    
    unet = UNET_3D_to_2D(0,channels_in=1,channels=64, growth_rate=16, dilated_layers=[4,4,4,4], output_channels=len(target_label_names)).to(device)
    unet.load_state_dict(checkpoint['state_dict'])
    unet.eval()

    t1 = nib.load(t1_file)
    t1_data = t1.get_fdata(dtype=np.float32)
    validate_input(t1, t1_data)
    
    # apply model
    with torch.set_grad_enabled(False):
        logit = apply_to_case(unet, volumes = [t1_data], batch_size=BATCH_SIZE, stack_depth = stack_depth, axes=[0,1,2], lowmem=args.lowmem)

    print('DONE predicting') if VERBOSE else False
    
    brain_tissue = logit[-1] > 0
    logit_sm = np.concatenate([logit[:-NUM_IGNORE_LABELS]], axis = 0)
    segmentation_sm = np.argmax(logit_sm, axis=0) + 1
    segmentation_sm_masked = segmentation_sm * brain_tissue
    segmentation_stacked  = np.stack([segmentation_sm_masked == x for x in range(1, len(target_label_names)-NUM_IGNORE_LABELS+1)], axis=0)
    
    volumes = np.sum(segmentation_stacked, axis=(1,2,3))
    
    # re-label with FS labels
    segmentation_sm_fslabels = np.zeros_like(segmentation_sm_masked)
    for idx in range(1, len(target_label_names)):
        segmentation_sm_fslabels[segmentation_sm_masked == idx] = LABELS[target_label_names[idx-1]]
    
    affine = t1.affine
    nib.save(nib.Nifti1Image(segmentation_sm_fslabels.astype(np.int32), affine), '{}/softmax_seg.nii.gz'.format(output_dir))
    
    # save individual logits for each class
    for idx, x in enumerate(target_label_names):
        lbl_name = target_label_names[idx]
        if not SAVE_LOGITS_FILTER or lbl_name in SAVE_LOGITS_FILTER:
            nib.save(nib.Nifti1Image(logit[idx, :, :, :].astype(np.float32), affine), '{}/seg_{}.nii.gz'.format(output_dir, lbl_name))

    with open('{}/result-vol.csv'.format(output_dir), 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['SUBJECT'] + ['{}'.format(target_label_names[idx]) for idx in range(0, len(target_label_names)-NUM_IGNORE_LABELS)])
        writer.writerow([subject_id] + ['{}'.format(i) for i in volumes])
        
    # write label definitions
    pd.DataFrame([[LABELS[lbl], lbl] for lbl in LABELS], columns=['ID', 'LABEL']).to_csv('{}/label_def.csv'.format(output_dir), sep=',', index=False)

