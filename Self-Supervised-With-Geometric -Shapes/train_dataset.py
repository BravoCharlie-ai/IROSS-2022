import open3d as o3d
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch
from augment_created_frame import aug_created_frame

def create_frame(number_of_objects):
    selected_objects=[random.randint(1, 196) for i in range(number_of_objects)]
    all_points=[]
    points_split = [0]
    for i in selected_objects:
        number=str(i+1)
        x=random.randint(-100, -100)
        y=random.randint(-100, -100)
        #z=random.randint(-40, 40)/4
        z=random.randint(80, 100)/100
        filename='train/car_'+number.zfill(4)+'.off'
        mesh = o3d.io.read_triangle_mesh(filename)
        number_of_points=random.randint(100, 3000)
        points=mesh.sample_points_uniformly(number_of_points=number_of_points)
        center=np.asarray(mesh.get_center())
        points=np.asarray(points.points)

        scale=random.randint(250, 400)/100

        points=points-center
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        extent=pcd.get_oriented_bounding_box().extent
        extent=1/extent
        extent=[extent[0],extent[0],extent[0]]
        points=points*extent
        points=points*[scale,scale,scale]
        points=points+[float(x/4),float(y/4),float(z)]

        points=aug_created_frame(points)
        
        points_split.append(points_split[-1]+points.shape[0])
        all_points.append(np.asarray(points))
    #all_points=np.concatenate(all_points, axis=0)    
    example={'points':all_points, 'points_split':points_split}
    return all_points
def get_aug_frame(all_points):
    bbox_points=[]
    points_split = [0]
    for points in all_points:
        random_translate_x = random.choice([0.1,0.2,0.4])
        random_translate_y = random.choice([0.1,0.2,0.4])
        random_translate_z = random.choice([0.1,0.2,0.4])
        points=points+[random_translate_x,random_translate_y,random_translate_z]
        bbox_points.append(points)
        points_split.append(points_split[-1]+points.shape[0])
    bbox_points = np.concatenate(bbox_points, axis=0)
    example = {
        'points': bbox_points,
        'points_split': points_split,
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
	})
	return frame_det_info

class TrainDataset(Dataset):
    def __init__(self,total_length):
        self.total_length=total_length
        pass
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        return self._generate_lidar()
    
    def _generate_lidar(self):
        number_of_objects=random.randint(10, 50)
        #number_of_objects=2
        all_points=create_frame(number_of_objects)
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
            point_cloud=get_aug_frame(all_points)
            det_num=number_of_objects
            det_split.append(torch.LongTensor([det_num]))
            det_info['points'].append(torch.Tensor(point_cloud['points']))
            det_info['points_split'].append(torch.Tensor(point_cloud['points_split'])[first_flag:])
            det_ids.append(det_id)
            det_cls.append(det_cl)
            if first_flag == 0:
                first_flag += 1

        det_imgs = []
        det_info['points'] = torch.cat(det_info['points'], dim=0)
        det_info['points']=torch.unsqueeze(det_info['points'], 0)
        start=0
        for i in range(len(det_info['points_split'])):
            det_info['points_split'][i] += start
            start = det_info['points_split'][i][-1]
        det_info['points_split'] = torch.cat(det_info['points_split'], dim=0)
        return det_imgs, det_info, det_ids, det_cls, det_split

    
    