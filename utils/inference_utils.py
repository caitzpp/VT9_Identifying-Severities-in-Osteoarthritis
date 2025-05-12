#dataloader
## need filepaths
import torch
from utils import create_mat_patches
import h5py #need to save it as very large file

def ss_create_rs(file_path, model_path, save_path, device, seed = 1001, bs = 1):
    test_dataset = 
    train_dataset = 

    loader = torch.utils.DataLoader(test_dataset, batch_size = bs, shuffle = False, num_workers=1, drop_last=False)
    

    model = ALEXNET_nomax_pre().to(device)
    model.load_state_dict(torch.load(model_path))
    
    model.eval()

    with torch.no_grad():
        for images, filenames in dataloader:

            #create feature_map & save it
            output = model.forward(img.float())

            #save direct feature extraction from AlexNet ! then we don't have to run through it again.

            #create patches and then use pooling layer:
            output = create_patches(output, padding, patchsize, stride)
            output=F.adaptive_avg_pool2d(output, (1,1) )[:,:,0,0].squeeze(1)
        



    
    

#patches, args.padding,args.patchsize, args.stride,seed, train_dataset, test_dataset, model, args.data_path, criterion, args.device, shots, args.meta_data_dir, args.get_oarsi_results


def ss_training(args, model_temp_name_ss, N, epochs, num_ss, shots, self_supervised, semi, seed = None, eval_epoch = 1): #trains the model and evaluates every 10 epochs for all seeds OR trains the model for a specific number of epochs for specified seed

  print("Trying to load val_dataset")
  val_dataset =  oa(args.data_path, task = 'test_on_train', train_info_path = args.train_ids_path)
  print("Val Dataset Loaded")
  if args.ss_test:
      test_dataset =  oa(args.data_path, task = args.task)
      #print("ss_test working, test_dataset loaded")
  else:
      test_dataset = None

  if seed == None:
     seeds =[1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
  else:
      seeds =[seed]

  current_epoch = 0
  for seed in seeds:
      model = ALEXNET_nomax_pre().to(args.device)
      train_dataset =  oa(args.data_path, task='train', stage='ss', N = N, shots = shots, semi = semi, self_supervised = self_supervised, num_ss = num_ss, augmentations = args.augmentations, normal_augs = args.normal_augs, train_info_path = args.train_ids_path, seed = seed)
      print(f"Training with {len(train_dataset)} samples")
      #print(f"First 5 paths:" {train_dataset.paths[:5]})
      train(train_dataset, val_dataset, N, model, epochs, seed, eval_epoch, shots, model_name_temp_ss + '_seed_' + str(seed), args, current_epoch, metric='centre_mean', patches =True, test_dataset = test_dataset )
      print("Training Done")
      del model
      print("Model Deleted")

  return os.path.join(args.dir_path, 'outputs/dfs/ss/'), os.path.join(args.dir_path, 'outputs/logs/ss/')

