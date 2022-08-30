import torch
import torch.nn as nn
import torch.nn.functional as F


class box_features(nn.Module):

    def __init__(self):
        super(box_features, self).__init__()
        # an affine operation: y = Wx + b
        '''
        self.fc1 = nn.Linear(7, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 240)
        self.fc3 = nn.Linear(240, 240)
        self.fc4 = nn.Linear(240, 512)'''

        self.fc1 = nn.Conv2d(7, 32, kernel_size=1, padding=0, bias=False)  # 5*5 from image dimension
        self.fc2 = nn.Conv2d(32, 32, kernel_size=1, padding=0, bias=False)
        self.fc3 = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False)
        self.fc4 = nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x=x.transpose(-1, -2)
        x=x.unsqueeze(-1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x=x.squeeze(-1)
        x=x.transpose(-1, -2)
        return x

    '''

    box_feature_net=box_features().cuda()
    features=[]
    def get_features(boxes,max_object):
        for frame_boxes in boxes:
            features.append(box_feature_net(frame_boxes.cuda()))

        frame_1_box_features=features[0][0]
        m = nn.ConstantPad2d((0, 0, 0, max_object-frame_1_box_features.shape[0]), 0)
        frame_1_box_features=m(frame_1_box_features)
        frame_1_box_features=torch.unsqueeze(frame_1_box_features, 0)


        frame_2_box_features=features[1][0]
        m = nn.ConstantPad2d((0, 0, 0, max_object-frame_2_box_features.shape[0]), 0)
        frame_2_box_features=m(frame_2_box_features)
        frame_2_box_features=torch.unsqueeze(frame_2_box_features, 0)

        return frame_1_box_features , frame_2_box_features'''
