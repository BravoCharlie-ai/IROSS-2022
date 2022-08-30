import open3d as o3d
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch
def aug_created_frame(points):
    pcd_for_rotation = o3d.geometry.PointCloud()
    pcd_for_rotation.points = o3d.utility.Vector3dVector(points)
    center=pcd_for_rotation.get_center()
    #rot_angle=random.choice([np.pi/1,np.pi/2,np.pi/3,np.pi/4,np.pi/4,np.pi/5, np.pi/6, -np.pi/1,-np.pi/2,-np.pi/3,-np.pi/4,-np.pi/4,-np.pi/5, -np.pi/6])
    rot_angle=random.choice([np.pi/8,0,0,-np.pi/8])
    rotate_object_z = np.array([
        [np.cos(rot_angle), np.sin(rot_angle), 0.0],
        [-np.sin(rot_angle), np.cos(rot_angle), 0.0],
        [0.0, 0.0, 1.0]
        ])
    pcd_for_rotation.rotate(rotate_object_z, center)
    points=np.asarray(pcd_for_rotation.points)
    
    lidar=points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    oriented_bbox=pcd.get_oriented_bounding_box()
    
    max_1=oriented_bbox.get_max_bound()
    min_1=oriented_bbox.get_min_bound()
    [maxX,maxY,maxZ]=max_1
    [minX,minY,minZ]=min_1
    
    number_cut_outs=random.choice([0,1,2,2,2,3])
    for k in range(number_cut_outs):
        random_cutout=random.choice(['left_1_a','left_1_b','left_2_a','left_2_b','right_1_a','right_1_b','right_2_a','right_2_b','z_axis_a','z_axis_b'])
        if random_cutout=='left_1_a':
            minX_2=float(minX+((maxX-minX)/8))
            #minY_2=float(minY+((maxY-minY)/8))
            mask = np.where((lidar[:, 0] >= minX_2) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
            #print('left_1_a')
        elif random_cutout=='left_1_b':
            minX_2=float(minX+((maxX-minX)/4))
            #minY_2=float(minY+((maxY-minY)/8))
            mask = np.where((lidar[:, 0] >= minX_2) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
            #print('left_1_b')
        elif random_cutout=='left_2_a':
            #minX_2=float(minX+((maxX-minX)/8))
            minY_2=float(minY+((maxY-minY)/6))
            mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY_2) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
            #print('left_2_a')
        elif random_cutout=='left_2_b':
            #minX_2=float(minX+((maxX-minX)/8))
            minY_2=float(minY+((maxY-minY)/4))
            mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY_2) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
            #print('left_2_b') 

        elif random_cutout=='right_1_a':
            #maxX_2=float(maxX-((maxX-minX)/8))
            maxY_2=float(maxY-((maxY-minY)/6))
            mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY_2) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
            #print('right_1_a')
            
        elif random_cutout=='right_1_b':
            #maxX_2=float(maxX-((maxX-minX)/8))
            maxY_2=float(maxY-((maxY-minY)/4))
            mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY_2) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
            #print('right_1_b')
            
        elif random_cutout=='right_2_a':
            maxX_2=float(maxX-((maxX-minX)/6))
            #maxY_2=float(maxY-((maxY-minY)/8))
            mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX_2) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
            #print('right_2_a')
            
        elif random_cutout=='right_2_b':
            maxX_2=float(maxX-((maxX-minX)/4))
            #maxY_2=float(maxY-((maxY-minY)/8))
            mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX_2) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
            #print('right_2_b')
            
        elif random_cutout=='z_axis_a':
            minZ_2=float(minZ+((maxZ-minZ)/4))
            mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ_2) & (lidar[:, 2] <= maxZ))
            #print('z_axis_a')
        elif random_cutout=='z_axis_b':
            minZ_2=float(minZ+((maxZ-minZ)/6))
            mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ_2) & (lidar[:, 2] <= maxZ))
            #print('z_axis_b')
            
            
        lidar=lidar[mask]
        
    points=lidar
    pcd_for_rotation = o3d.geometry.PointCloud()
    pcd_for_rotation.points = o3d.utility.Vector3dVector(points)
    center=pcd_for_rotation.get_center()
    rot_angle=random.choice([np.pi/1,np.pi/2,np.pi/3,np.pi/4,np.pi/4,np.pi/5, np.pi/6, -np.pi/1,-np.pi/2,-np.pi/3,-np.pi/4,-np.pi/4,-np.pi/5, -np.pi/6])
    #rot_angle=random.choice([np.pi/8,0,0,-np.pi/8])
    rotate_object_z = np.array([
        [np.cos(rot_angle), np.sin(rot_angle), 0.0],
        [-np.sin(rot_angle), np.cos(rot_angle), 0.0],
        [0.0, 0.0, 1.0]
        ])
    pcd_for_rotation.rotate(rotate_object_z, center)
    points=np.asarray(pcd_for_rotation.points)
    return points
    