import torch
from smplx import build_layer as build_body_model

class SMPLXLayer(object):
    def __init__(self, cfg,
                pose_desc_dict,
                body_model_path = 'data/models/', 
                model_type = 'smplx',
                dtype = torch.float32):
        self.pose_desc_dict = pose_desc_dict
        self.body_model = build_body_model(
                        body_model_path,
                        model_type=model_type,
                        dtype=dtype, **cfg).cuda()
        self.faces = self.body_model.faces
        self.get_param_mean(pose_desc_dict)
        self.get_idxs(pose_desc_dict)
    def get_idxs(self, pose_desc_dict, 
                    num_betas = 10, 
                    num_expression_coeffs = 10, 
                    cam_param_num = 3, dtype = torch.long):
        start = 0 
        #get global orient idxs
        global_orient_dim = pose_desc_dict['global_orient']['dim']
        self.global_orient_idxs = torch.tensor(list(range(start, start + global_orient_dim)), dtype= dtype).cuda()
        start += global_orient_dim
        #get body pose idxs
        body_pose_dim = pose_desc_dict['body_pose']['dim']
        self.body_pose_idxs = torch.tensor(list(range(start, start + body_pose_dim)), dtype= dtype).cuda()
        start += body_pose_dim
        #get left hand pose idxs
        left_hand_pose_dim = pose_desc_dict['left_hand_pose']['dim']
        self.left_hand_pose_idxs = torch.tensor(list(range(start, start + left_hand_pose_dim)), dtype= dtype).cuda()
        start += left_hand_pose_dim
        #get left hand pose idxs
        right_hand_pose_dim = pose_desc_dict['right_hand_pose']['dim']
        self.right_hand_pose_idxs = torch.tensor(list(range(start, start + right_hand_pose_dim)), dtype= dtype).cuda()
        start += right_hand_pose_dim
        #get jaw pose idxs
        jaw_pose_dim = pose_desc_dict['jaw_pose']['dim']
        self.jaw_pose_idxs = torch.tensor(list(range(start, start + jaw_pose_dim)), dtype= dtype).cuda()
        start += jaw_pose_dim
        #additional for shape, expression, camera parameter
        #shape
        shape_dim = num_betas
        self.shape_idxs = torch.tensor(list(range(start, start + shape_dim)), dtype= dtype).cuda()
        start += shape_dim
        #expression
        exp_dim = num_expression_coeffs
        self.expression_idxs = torch.tensor(list(range(start, start + exp_dim)), dtype= dtype).cuda()
        start += exp_dim
        #camera 
        cam_dim = cam_param_num
        self.camera_idxs = torch.tensor(list(range(start, start + cam_dim)), dtype= dtype).cuda()
        start += cam_dim
    def get_param_mean(self, pose_desc_dict):
        self.mean_list = []
        global_orient_mean = pose_desc_dict['global_orient']['mean']
        global_orient_mean[3] = -1

        body_pose_mean = pose_desc_dict['body_pose']['mean']
        left_hand_pose_mean = pose_desc_dict['left_hand_pose']['mean']
        right_hand_pose_mean = pose_desc_dict['right_hand_pose']['mean']
        jaw_pose_mean = pose_desc_dict['jaw_pose']['mean']
        shape_mean = pose_desc_dict['shape_mean']
        exp_mean = pose_desc_dict['exp_mean']
        camera_mean = pose_desc_dict['camera']['mean']

        self.param_mean = torch.cat([global_orient_mean, body_pose_mean, left_hand_pose_mean, right_hand_pose_mean, jaw_pose_mean, shape_mean, exp_mean, camera_mean]).view(1, -1)
    def flat_body_params_to_dict(self, param_tensor):
        '''EDITED change using cuda variable'''
        global_orient = torch.index_select(
            param_tensor, 1, self.global_orient_idxs)
        body_pose = torch.index_select(
            param_tensor, 1, self.body_pose_idxs)
        left_hand_pose = torch.index_select(
            param_tensor, 1, self.left_hand_pose_idxs)
        right_hand_pose = torch.index_select(
            param_tensor, 1, self.right_hand_pose_idxs)
        jaw_pose = torch.index_select(
            param_tensor, 1, self.jaw_pose_idxs)
        betas = torch.index_select(param_tensor, 1, self.shape_idxs)
        expression = torch.index_select(param_tensor, 1, self.expression_idxs)
        camera = torch.index_select(param_tensor, 1, self.camera_idxs )
        return {
            'betas': betas,
            'expression': expression,
            'global_orient': global_orient,
            'body_pose': body_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'jaw_pose': jaw_pose,
            'camera': camera
        } 
    def change_param_form(self, param):
        global_orient = self.pose_desc_dict['global_orient']['decoder'](param['global_orient'].clone())
        body_pose = self.pose_desc_dict['body_pose']['decoder'](param['body_pose'])
        left_hand_pose = self.pose_desc_dict['left_hand_pose']['decoder'](param['left_hand_pose'])
        right_hand_pose = self.pose_desc_dict['right_hand_pose']['decoder'](param['right_hand_pose'])
        jaw_pose = self.pose_desc_dict['jaw_pose']['decoder'](param['jaw_pose'])
        return {'betas': param['betas'],
                'expression': param['expression'],
                'global_orient': global_orient,
                'body_pose': body_pose,
                'left_hand_pose': left_hand_pose,
                'right_hand_pose': right_hand_pose,
                'jaw_pose': jaw_pose,
                'camera': param['camera']
        }
    def get_2d_pred(self, pred_joints, pred_param):
        scale = pred_param['camera'][:, 0].view(-1, 1)
        translation = pred_param['camera'][:, 1:3]
        proj_joints = self.pose_desc_dict['camera']['camera']\
            (pred_joints, scale = scale, translation = translation)
        return proj_joints
    def gen_smplx(self, input, name_type = 'gt'):
        if name_type == 'gt':
            body_model_output = self.body_model(get_skin=True, return_shaped=True, **input)
            return body_model_output
        else:
            param_dicts = []
            gt_list = []
            for i in range(3):
                curr_param = self.flat_body_params_to_dict(input[i])
                reform_pred = self.change_param_form(curr_param)
                param_dicts.append(reform_pred)
            last_param = param_dicts[-1]
            body_model_output = self.body_model(
                get_skin=True, return_shaped=True, **last_param)
            return body_model_output, last_param