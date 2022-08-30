import open3d as o3d
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch
import random
print('creating basic objects')

def create_frame(number_of_objects):
    all_points=[]
    all_boxes=[]
    points_split = [0]
    for i in range(number_of_objects):
        x=random.randint(-20, 20)
        y=random.randint(-20, 20)
        z=random.randint(80, 100)/100
        
        radius=random.random()*random.choice([0.5,1])
        height=random.random()*random.choice([1,2])
        resolution=random.randint(100,200)
        number_of_points=random.randint(100,500)
        
        mesh=o3d.geometry.TriangleMesh.create_cone(radius,height)        
        points=mesh.sample_points_uniformly(number_of_points, use_triangle_normal=False, seed=- 1)
        
        points=np.asarray(points.points)
        points=points+[float(x),float(y),float(z)]

        
        
        points_split.append(points_split[-1]+points.shape[0])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        get_box=pcd.get_oriented_bounding_box()
        get_box=list(get_box.center)+list(get_box.extent)+[0]

        augment_rotation=True
        if augment_rotation:

            pcd_for_rotation = o3d.geometry.PointCloud()
            pcd_for_rotation.points = o3d.utility.Vector3dVector(points[:,0:3])
            center=pcd_for_rotation.get_center()
            rot_angle=random.random()*np.pi*random.choice([1,-1])
            '''
            rotate_object_z = np.array([
                [np.cos(rot_angle), np.sin(rot_angle), 0.0],
                [-np.sin(rot_angle), np.cos(rot_angle), 0.0],
                [0.0, 0.0, 1.0]
                ])'''
            
            rotate_object_z = np.array([
                [np.cos(rot_angle), 0.0,  np.sin(rot_angle)],
                [0.0, 1.0 , 0.0],
                [-np.sin(rot_angle), 0.0,  np.cos(rot_angle)]
                ])
            pcd_for_rotation.rotate(rotate_object_z, center)
            points[:,0:3]=np.asarray(pcd_for_rotation.points)

            get_box[6]=get_box[6]+np.asarray([rot_angle])

        all_boxes.append(get_box)


        all_points.append(np.asarray(points))
    #all_points=np.concatenate(all_points, axis=0)    
    example={'points':all_points, 'points_split':points_split}
    return all_points, all_boxes

def get_aug_frame(all_points,all_boxes,points_per_object):
    #print('all_boxes', len(all_boxes))
    #print('all_points', len(all_points))
    bbox_points=[]
    bbox_points_scaled=[]
    box3D=[]
    points_split = [0]
    for i in range(len(all_points)):
        #for points in all_points:
        points=all_points[i]
        get_box=all_boxes[i]

        lidar=points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        oriented_bbox=pcd.get_oriented_bounding_box()
        
        max_1=oriented_bbox.get_max_bound()
        min_1=oriented_bbox.get_min_bound()
        [maxX,maxY,maxZ]=max_1
        [minX,minY,minZ]=min_1
        random_cutout=random.choice(['left','right','None'])
        if random_cutout=='left':
            minX_2=float(minX+((maxX-minX)/4))
            minY_2=float(minY+((maxY-minY)/4))
            mask = np.where((lidar[:, 0] >= minX_2) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY_2) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))

        elif random_cutout=='right':
            maxX_2=float(maxX-((maxX-minX)/4))
            maxY_2=float(maxY-((maxY-minY)/3))
            mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX_2) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY_2) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))

        else:
            mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
            
        points=lidar[mask]

        random_translate_x = random.choice([0.1,0.2,0.4])
        random_translate_y = random.choice([0.1,0.2,0.4])
        random_translate_z = random.choice([0.1,0.2,0.4])

        points=points+[random_translate_x,random_translate_y,random_translate_z]

        get_box[0:3]=get_box[0:3]+np.asarray([random_translate_x,random_translate_y,random_translate_z])


        rotation_aug=True
        if rotation_aug:

            pcd_for_rotation = o3d.geometry.PointCloud()
            pcd_for_rotation.points = o3d.utility.Vector3dVector(points[:,0:3])
            center=pcd_for_rotation.get_center()
            rot_angle=random.choice([0, np.pi/36, np.pi/18, -np.pi/36, -np.pi/18])
            rotate_object_z = np.array([
                [np.cos(rot_angle), np.sin(rot_angle), 0.0],
                [-np.sin(rot_angle), np.cos(rot_angle), 0.0],
                [0.0, 0.0, 1.0]
                ])
            pcd_for_rotation.rotate(rotate_object_z, center)
            points[:,0:3]=np.asarray(pcd_for_rotation.points)

            get_box[6]=get_box[6]+np.asarray([rot_angle])


        box3D.append(get_box)
        bbox_points.append(points)
        choice = np.random.choice(len(points), points_per_object, replace=True)
        bbox_points_scaled.append(points[:,0:3][choice, :])
        #print('bbox_points_scaled', points[:,0:3][choice, :].shape)
        points_split.append(points_split[-1]+points.shape[0])



    bbox_points_scaled = np.concatenate([bbox_points_scaled], axis=0)
    bbox_points_scaled=np.einsum('ijk->kji', bbox_points_scaled)
    #print(bbox_points_scaled.shape)
    bbox_points = np.concatenate(bbox_points, axis=0)
    example = {
        'points': bbox_points,
        'points_split': points_split,
        'bbox_points_scaled' : bbox_points_scaled,
        'boxes': box3D,
        }
    
    return example
def get_frame_det_info():
	frame_det_info = {}
	frame_det_info.update({
		'rot': [],
		'loc': [],
		'dim': [],
		'points': [],
		'points_split': [],
		'info_id': [],
        'bbox_points_scaled' : [],
        'boxes' : [],
	})
	return frame_det_info

class TrainDataset(Dataset):
    def __init__(self,total_length,points_per_object=512):
        self.total_length=total_length
        self.points_per_object=points_per_object
        pass
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        return self._generate_lidar()
    
    def _generate_lidar(self):
        number_of_objects=random.randint(5, 100)
        #number_of_objects=2
        all_points, all_boxes=create_frame(number_of_objects)
        det_imgs = []
        det_split = []
        dets = []
        det_cls = []
        det_ids = []
        det_info = get_frame_det_info()
        first_flag = 0
        det_id=[id for id in range(number_of_objects)]
        det_cl=[1 for j in range(number_of_objects)]
        det_id=torch.Tensor(det_id)
        det_id=torch.unsqueeze(det_id, 0)
        det_id=torch.unsqueeze(det_id, -1)

        det_cl=torch.Tensor(det_cl)
        det_cl=torch.unsqueeze(det_cl, 0)
        det_cl=torch.unsqueeze(det_cl, -1)

        for i in range (2):
            point_cloud=get_aug_frame(all_points,all_boxes,self.points_per_object)
            det_num=number_of_objects
            det_split.append(torch.LongTensor([det_num]))
            det_info['points'].append(torch.Tensor(point_cloud['points']))
            det_info['points_split'].append(torch.Tensor(point_cloud['points_split'])[first_flag:])
            det_info['bbox_points_scaled'].append(torch.Tensor(point_cloud['bbox_points_scaled']))
            #print('bbox_points_scaled',torch.Tensor(point_cloud['bbox_points_scaled']).shape)
            det_info['boxes'].append(torch.Tensor(point_cloud['boxes']).unsqueeze(0))



            det_ids.append(det_id)
            det_cls.append(det_cl)
            if first_flag == 0:
                first_flag += 1

        det_imgs = []
        det_info['points'] = torch.cat(det_info['points'], dim=0)
        det_info['points']=torch.unsqueeze(det_info['points'], 0)
        det_info['bbox_points_scaled'] = torch.cat(det_info['bbox_points_scaled'], dim=-1)
        det_info['bbox_points_scaled']=torch.unsqueeze(det_info['bbox_points_scaled'], 0)
        #print(det_info['bbox_points_scaled'].shape)
        start=0
        for i in range(len(det_info['points_split'])):
            det_info['points_split'][i] += start
            start = det_info['points_split'][i][-1]
        det_info['points_split'] = torch.cat(det_info['points_split'], dim=0)
        return det_imgs, det_info, det_ids, det_cls, det_split

    
    