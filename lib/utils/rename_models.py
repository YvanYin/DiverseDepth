import torch
import dill


weight_path = '/home/yvan/DeepLearning/Depth/depth-2.3-ASPP-V3.2/models/kitti-benchmark/epoch1_step144000.pth'
output_path = '/'.join(weight_path.split('/')[0:-1]) + '/final_' + weight_path.split('/')[-1]

checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage, pickle_module=dill)
src_dict = checkpoint['model_state_dict']
dst_dict = {}
for k, v in src_dict.items():
    if 'depth_normal_model.lateral_modules' in k:
        k = k.replace('depth_normal_model.lateral_modules', 'depth_model.encoder_modules')
    if 'depth_normal_model.topdown_modules' in k:
        k = k.replace('depth_normal_model.topdown_modules', 'depth_model.decoder_modules')
    dst_dict[k] = v
checkpoint['model_state_dict'] = dst_dict

#torch.save(checkpoint, output_path, pickle_module=dill)
ck_dict = {}
ck_dict['model_state_dict'] = dst_dict
torch.save(ck_dict, output_path, pickle_module=dill)

print('finish')