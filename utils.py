import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
import torch.nn.functional as F

def init_regul(source_vertices, source_faces):
    sommet_A_source = source_vertices[source_faces[:, 0]]
    sommet_B_source = source_vertices[source_faces[:, 1]]
    sommet_C_source = source_vertices[source_faces[:, 2]]
    target = []
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    # print(len(target))
    return target

def get_target(vertice, face, size):
    target = init_regul(vertice,face)
    target = np.array(target)
    target = torch.from_numpy(target).float().cuda()
    #target = target+0.0001
    target = target.unsqueeze(1).expand(3,size,-1)
    return target

def calc_euclidean_dist_matrix(x):
    #OH: x contains the coordinates of the mesh,
    #x dimensions are [batch_size x num_nodes x 3]

    #x = x.transpose(2,1)
    r = torch.sum(x ** 2, dim=2).unsqueeze(2)  # OH: [batch_size  x num_points x 1]
    r_t = r.transpose(2, 1) # OH: [batch_size x 1 x num_points]
    inner = torch.bmm(x,x.transpose(2, 1))
    D = F.relu(r - 2 * inner + r_t)**0.5  # OH: the residual numerical error can be negative ~1e-16
    return D
def compute_score(points, faces, target):
    score = 0
    sommet_A = points[:,faces[:, 0]]
    sommet_B = points[:,faces[:, 1]]
    sommet_C = points[:,faces[:, 2]]

    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / target[0] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) / target[1] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / target[2] -1)

    return torch.mean(score)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)



def central_distance_mean_score(points, gt_points, faces):
    score = 0
    # print(points.shape)
    # print(gt_points.shape)

    for point_index in range(len(points)):
        # print(point_index)
        connected_trianlges = np.where(faces == point_index)[0]
        # print(connected_trianlges.shape)
        # print(connected_trianlges)
        # print(np.unique(faces[connected_trianlges,:]))
        connected_points_index = np.unique(faces[connected_trianlges,:])
        # print(connected_points_index)
        connected_points= points[connected_points_index]
        gt_connected_points= gt_points[connected_points_index]
        # print(connected_points.shape)
        # print(gt_connected_points.shape)

        current_point_array = points[point_index].repeat(connected_points.shape[0], 1)
        gt_current_point_array = gt_points[point_index].repeat(connected_points.shape[0], 1)
        # print(current_point_array.shape)
        # print(gt_current_point_array.shape)
        distance = connected_points - current_point_array
        gt_distance = gt_connected_points - gt_current_point_array
        loss = nn.MSELoss()
        score += loss(distance, gt_distance)

    return torch.mean(score)

def central_distance_gradient_score(points, gt_points, faces):
    score = 0

    for point_index in range(len(points)):
        # print(point_index)
        connected_trianlges = np.where(faces == point_index)[0]
        # print(connected_trianlges.shape)
        # print(connected_trianlges)
        connected_points_index = np.delete(np.unique(faces[connected_trianlges,:]), point_index)
        connected_points= points[:,connected_points_index]
        gt_connected_points= gt_points[:,connected_points_index]
        # print(connected_points)
        # print(connected_points.shape)connected_points
        current_point_array = points[:,point_index].repeat(connected_points.shape[1], 1)
        gt_current_point_array = gt_points[:,point_index].repeat(connected_points.shape[1], 1)
        # print(current_point_array)

        gt_distance

        score += torch.mean( torch.sqrt(torch.sum((connected_points - current_point_array) ** 2, dim=2)))

    return torch.mean(score)
