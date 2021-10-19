import torch.utils.data as data
import torch
import numpy as np
import trimesh
from objLoader_trimesh import trimesh_load_obj
# from objLoader_trimesh_animal import trimesh_load_obj_animal
import random

class SMPL_sequence(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400, video_len = 1):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./datasets/DFAUST_sequence_data_all/'
        self.length = training_size
        self.video_len = video_len

    def __getitem__(self, index):
        if self.train:
         
            identity_mesh_i=np.random.randint(0,16)#1-16train. rest test,
            identity_mesh_p=np.random.randint(0,80)#1-80train. rest test,
            identity_mesh_fr=np.random.randint(0,30)

            pose_mesh_i=np.random.randint(0,16)
            pose_mesh_p=np.random.randint(0,80)
            pose_mesh_fr=np.random.randint(0,30-self.video_len)

        '''load target mesh'''
       
        identity_mesh_path = self.path+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'_'+str(identity_mesh_fr)+'.obj'
        #print('target mesh')
        #print(identity_mesh_path)
        identity_mesh=trimesh_load_obj(identity_mesh_path)
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        #pose_mesh_sequence
        '''load pose and gt sequence meshes'''
        # print('length')
        # print(self.video_len)
        pose_mesh_sequence = np.zeros((self.video_len, self.npoints, 3))
        gt_mesh_sequence = np.zeros((self.video_len, self.npoints,3 ))
        idx = 0
        for frame in range(pose_mesh_fr,pose_mesh_fr + self.video_len):
            pose_mesh_path =self.path+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'_'+str(frame)+'.obj'
            gt_mesh_path = self.path+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'_'+str(frame)+'.obj'
            #print('source mesh')
            #print(pose_mesh_path)
            #print(gt_mesh_path)

            pose_mesh=trimesh_load_obj(pose_mesh_path)
            gt_mesh=trimesh_load_obj(gt_mesh_path)

            pose_points = pose_mesh.vertices
            gt_points = gt_mesh.vertices

            pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
            gt_points = gt_points - (gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
            # print(pose_points.shape)
            pose_mesh_sequence[idx,:,:] = pose_points
            gt_mesh_sequence[idx,:,:]  = gt_points
            idx +=1

        pose_mesh_sequence = torch.from_numpy(pose_mesh_sequence.astype(np.float32))
        gt_mesh_sequence = torch.from_numpy(gt_mesh_sequence.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces
        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)
        return pose_mesh_sequence, gt_mesh_sequence, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)


