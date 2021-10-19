import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import SMPL_sequence,SMG_DATA,SMAL_DATA
import utils as utils
import numpy as np
import time,cv2
import trimesh
from tqdm import tqdm
import argparse
from weak_perspective_pyrender_renderer import Renderer
parser = argparse.ArgumentParser(description='Training NTP parameters')
parser.add_argument('--batch_size', type=int,default=8,help='training batch size')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle mesh points')
parser.add_argument('--model_type', type=str,default='original',help='model type')
parser.add_argument('--train_epoch', type=int,default=200,help='training epoch')
parser.add_argument('--train_size', type=int,default=400,help='training data size')
parser.add_argument('--dataset_name', type=str,default='SMG-3D',help='training data set')
parser.add_argument('--keep_train', type=int,default=0, help='keep training from checkpoint')
parser.add_argument('--lamda', type=float,default=0.0, help='center loss')
parser.add_argument('--video_len', type=int,default=1, help='video len')

args = parser.parse_args()

batch_size = args.batch_size
shuffle_point = args.shuffle
train_epoch = args.train_epoch
train_size = args.train_size
dataset_name = args.dataset_name
keep_train = args.keep_train
lamda = args.lamda
video_len = args.video_len

model_type = args.model_type
if model_type == 'model_3Dtransformer_basic':
    from model.model_3Dtransformer_basic import Transformer3D
elif model_type == 'model_3Dtransformer_full':
    from model.model_3Dtransformer_frameseperate import Transformer3D
elif model_type == 'model_3Dtransformer_temporal_embedding_mlp':
    from model.model_3Dtransformer_temporal_embedding_mlp import Transformer3D
elif model_type == 'model_3Dtransformer_temporal_embedding':
    from model.model_3Dtransformer_temporal_embedding import Transformer3D
elif model_type == 'CGP':
    from model.model_CGP import NPT
else:
    print('wrong model')

if dataset_name =='SMPL-sequence':
    dataset = SMPL_sequence(train=True, shuffle_point = shuffle_point, training_size = train_size, video_len = args.video_len)
elif dataset_name =='SMAL':
    dataset = SMAL_DATA(train=True, shuffle_point = shuffle_point, training_size = train_size)
else:
    print('wrong dataset')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
# dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)

model=Transformer3D(num_points = 6890, bottleneck_size = 1024, video_len = video_len)
lrate=0.00005

if torch.cuda.device_count() > 1:
    print('multi gpu')
    print(torch.cuda.device_count())
    model = torch.nn.DataParallel(model)

model.cuda()

optimizer_G = optim.Adam(model.parameters(), lr=lrate)

print(keep_train)
if keep_train:
    checkpoint_path='./saved_model/'+dataset_name+'_type'+model_type+'_sf'+str(shuffle_point)+'_bs'+str(batch_size)+'_ts' + str(train_size) +'_ep'+ str(train_epoch)+'_lamda_'+str(lamda)+'.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print('Keeping training from epoch: ' + str(start_epoch))
else:
    model.apply(utils.weights_init)
    start_epoch = 0

wp_renderer = Renderer(resolution=(640, 640))
scheduler = MultiStepLR(optimizer_G, milestones=[100,200,300], gamma=0.1)

print('training start')
print('Dataset:' + dataset_name)
print('Model:' + model_type)
print('Epoch:' + str(train_epoch))
print('Batch size:' + str(batch_size))
print('Sample size:' + str(train_size))
print('Shuffle point:' + str(shuffle_point))
print('Center loss Lamda:' + str(lamda))
loss_best = 0.2
torch.set_default_tensor_type(torch.FloatTensor)
for epoch in tqdm(range(start_epoch, train_epoch)):

    start=time.time()
    total_loss=0
    # switch model to evaluation mode
    model.train();
    '''training phase'''
    for j,data in enumerate(dataloader,0):

        optimizer_G.zero_grad()

        pose_mesh_sequence, gt_mesh_sequence, identity_points, new_face=data
        b, f, c, p = pose_mesh_sequence.shape

        #print(pose_mesh_sequence.shape)
        # print(pose_mesh_sequence.shape)
        # print(gt_mesh_sequence.shape)
        # print(identity_points.shape)
        # print(new_face.shape)

        pose_mesh_sequence=pose_mesh_sequence.transpose(3,2)
        pose_mesh_sequence=pose_mesh_sequence.cuda()

        identity_points=identity_points.transpose(2,1)
        identity_points=identity_points.cuda()

        gt_mesh_sequence=gt_mesh_sequence.cuda()

        pointsReconstructed_sequence = model(pose_mesh_sequence,identity_points)

        pointsReconstructed_sequence = pointsReconstructed_sequence.float()

        rec_loss = torch.mean((pointsReconstructed_sequence - gt_mesh_sequence)**2)
        # print('rec_loss')
        # print(rec_loss)
        motion_loss= 0
        for f_i in range(2,f):
            #motion_loss = ((pointsReconstructed_sequence[:,f_i,:,:] - pointsReconstructed_sequence[:,f_i-1,:,:])/(gt_mesh_sequence[:,f_i,:,:] - gt_mesh_sequence[:,f_i-1,:,:]))**2 - ((pointsReconstructed_sequence[:,f_i-1,:,:] - pointsReconstructed_sequence[:,f_i-2,:,:])/(gt_mesh_sequence[:,f_i-1,:,:] - gt_mesh_sequence[:,f_i-2,:,:]))**2
            motion_loss=motion_loss+torch.mean(((pointsReconstructed_sequence[:,f_i,:,:] - pointsReconstructed_sequence[:,f_i-1,:,:])-(gt_mesh_sequence[:,f_i,:,:] - gt_mesh_sequence[:,f_i-1,:,:]))**2)

        edg_loss= 0
        for b_i in range(b):
            face=new_face[0].cpu().numpy()
            
            for f_i in range(f):
                # print(v.shape)
                v=gt_mesh_sequence[b_i,f_i].cpu().numpy()
                #print(v.shape)
                edg_loss=edg_loss+utils.compute_score(pointsReconstructed_sequence[b_i,f_i].unsqueeze(0),face,utils.get_target(v,face,1))
        edg_loss=edg_loss/(b*f)
        # print('edg_loss')
        # print(edg_loss)

        # central_distance_loss= 0
        # for i in range(len(random_sample)):
        #     f=new_face[i].cpu().numpy()
        #     # print(f.shape)#(13776, 3)
        #     v=gt_points[i].unsqueeze(0)
        #     # print(v.shape)#(1,6890, 3)
        #     central_distance_loss += utils.central_distance_mean_score(pointsReconstructed_sequence[i].unsqueeze(0),v,f)
        # central_distance_loss=central_distance_loss/len(random_sample)

        # print('central_distance_loss')
        # print(central_distance_loss)
        # print(a)

        rec_loss=rec_loss+0.0005*edg_loss+lamda*motion_loss
        l2_loss=rec_loss
        # rec_loss=rec_loss+0.0005*edg_loss+lamda*central_distance_loss
        rec_loss.backward()
        optimizer_G.step()
        total_loss=total_loss+l2_loss

    print('####################################')
    # print(len(dataloader))
    print('Training')
    print('Epoch: ' +str(epoch))
    # print(time.time()-start)
    mean_loss=total_loss/(j+1)
    print('Mean_loss',mean_loss.item())
    scheduler.step()
    print('####################################')


    # print(optimizer_G.param_groups[0]['lr'])
    if loss_best>mean_loss.item():
        if keep_train:
            save_path='./saved_model/continue/'+dataset_name+'_type'+model_type+'_sf'+str(shuffle_point)+'_bs'+str(batch_size)+'_ts' + str(train_size) +'_ep'+str(train_epoch)+'_lamda_'+str(lamda)+'.model'
            checkpoint_path='./saved_model/continue/'+dataset_name+'_type'+model_type+'_sf'+str(shuffle_point)+'_bs'+str(batch_size)+'_ts' + str(train_size) +'_ep'+str(train_epoch)+'_lamda_'+str(lamda)+'.pt'
        else:
            save_path='./saved_model/'+dataset_name+'_type'+model_type+'_sf'+str(shuffle_point)+'_bs'+str(batch_size)+'_ts' + str(train_size) +'_ep'+str(train_epoch)+'_lamda_'+str(lamda)+'.model'
            checkpoint_path='./saved_model/'+dataset_name+'_type'+model_type+'_sf'+str(shuffle_point)+'_bs'+str(batch_size)+'_ts' + str(train_size) +'_ep'+str(train_epoch)+'_lamda_'+str(lamda)+'.pt'


        loss_best = mean_loss.item()

        torch.save(model.state_dict(),save_path)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict(),
                    'loss': rec_loss,
                    }, checkpoint_path)

        for frame in range(video_len):

            rend_img = wp_renderer.render(
            verts = pointsReconstructed_sequence[0][frame].detach().cpu().numpy(),
            faces = new_face[0].detach().cpu().numpy(),
            cam=np.array([0.8, 0., 0.2]),
            angle=-180,
            axis= [1, 0, 0])
            cv2.imwrite(f"./sample_3d_raw/{str(epoch).zfill(6)}_generated_" + str(frame)+".png", rend_img)
