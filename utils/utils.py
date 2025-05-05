import numpy as np
import torch
import torch.nn.functional as F

def create_patches(features, padding,patchsize, stride):
    n=False
    if type(features).__module__ == np.__name__:
        features = torch.FloatTensor(features)
        n=True

    unfolder = torch.nn.Unfold(
                kernel_size=patchsize, stride=stride, padding=padding, dilation=1
            )
    unfolded_features = unfolder(features)
    number_of_total_patches = []
    for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (patchsize - 1) - 1
            ) / stride + 1

            number_of_total_patches.append(int(n_patches))
            unfolded_features = unfolded_features.reshape(
            *features.shape[:2], patchsize, patchsize, -1
        )
    unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
    if n == True:
        unfolded_features = np.asarray(unfolded_features)


    return unfolded_features[0]

def create_mat_patches(ref_dataset, model, padding, patchsize, stride, dev, shots):
    for i in range(0, (len(ref_dataset.paths2)  )):
        img1, _, _, _,_,lab = ref_dataset.__getitem__(i)
        if i == 0:
            assert lab.item() == 0
            mat = F.adaptive_avg_pool2d(create_patches (model.forward( img1.to(dev).float()).detach(), padding, patchsize, stride) , (1,1) )[:,:,0,0].unsqueeze(0)
        else:
            if lab.item() == 0:
                mat = torch.cat((mat, F.adaptive_avg_pool2d( create_patches ( model.forward( img1.to(dev).float()).detach() , padding,patchsize, stride) , (1,1) )[:,:,0,0].unsqueeze(0) ))
            else:
                try:
                  mat_anom = torch.cat((mat_anom, F.adaptive_avg_pool2d( create_patches ( model.forward( img1.to(dev).float()).detach() , padding,patchsize, stride) , (1,1) )[:,:,0,0].unsqueeze(0) ))
                except:
                  mat_anom =  F.adaptive_avg_pool2d( create_patches ( model.forward( img1.to(dev).float()).detach() , padding,patchsize, stride) , (1,1) )[:,:,0,0].unsqueeze(0)

    if shots > 0:
        return mat, mat_anom
    else:
        return mat