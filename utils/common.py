import numpy as np
import torch
import pickle
import cv2
from .prior import ContinuousRotReprDecoder
from .camera import build_cam_proj

def load_all_mean_params(cfg, dtype=torch.float32):
    mean_poses_dict = {}
    PM_dir = cfg['common']['all_mean_dir']
    smplx_param = cfg['smplx']
    with open(PM_dir, 'rb') as f:
        mean_poses_dict = pickle.load(f)
    global_orient_desc = create_pose_param(smplx_param['num_global_orient'])
    body_pose_desc = create_pose_param(smplx_param['num_body_pose'], mean = mean_poses_dict['body_pose'])
    left_hand_pose_desc = create_pose_param(smplx_param['num_left_hand_pose'], mean = mean_poses_dict['left_hand_pose'])
    right_hand_pose_desc = create_pose_param(smplx_param['num_right_hand_pose'], mean = mean_poses_dict['left_hand_pose'])
    jaw_pose_desc = create_pose_param(smplx_param['num_jaw_pose'])

    shape_mean = torch.from_numpy(np.load(cfg['common']['shape_mean_dir'], allow_pickle=True)).to(
                dtype=dtype).reshape(1, -1)[:, :smplx_param['num_betas']].reshape(-1)
    expression_mean = torch.zeros([smplx_param['num_expression']], dtype=dtype)
    return {
        'global_orient': global_orient_desc,
        'body_pose': body_pose_desc,
        'left_hand_pose': left_hand_pose_desc,
        'right_hand_pose': right_hand_pose_desc,
        'jaw_pose': jaw_pose_desc,
        'shape_mean' : shape_mean, 
        'exp_mean' : expression_mean,
        'camera' : build_cam_proj()
    }
def get_param_mean(pose_desc_dict):
    mean_list = []
    global_orient_mean = pose_desc_dict['global_orient']['mean']
    global_orient_mean[3] = -1
    body_pose_mean = pose_desc_dict['body_pose']['mean']
    left_hand_pose_mean = pose_desc_dict['left_hand_pose']['mean']
    right_hand_pose_mean = pose_desc_dict['right_hand_pose']['mean']
    jaw_pose_mean = pose_desc_dict['jaw_pose']['mean']
    shape_mean = pose_desc_dict['shape_mean']
    exp_mean = pose_desc_dict['exp_mean']
    camera_mean = pose_desc_dict['camera']['mean']
    return torch.cat([global_orient_mean, body_pose_mean, left_hand_pose_mean, right_hand_pose_mean, jaw_pose_mean, shape_mean, exp_mean, camera_mean]).view(1, -1)
def create_pose_param(num_angles, mean = None):
    decoder = ContinuousRotReprDecoder(num_angles, mean = mean)
    dim = decoder.get_dim_size()
    ind_dim = 6
    mean = decoder.get_mean()
    return {
        'decoder': decoder,
        'dim' : dim,
        'ind_dim' : ind_dim,
        'mean' : mean,
    }
def pose_processing(pose, r, f = 0):
    """Process SMPL theta parameters  and apply all augmentation transforms."""
    # rotation or the pose parameters
    pose[:3] = rot_aa(pose[:3], r)
    # flip the pose parameters
    if f:
        pose = flip_pose(pose)
    # (72),float
    pose = pose.astype('float32')
    return pose
def get_dset_idxs( dset_name, exp_name):
    src_idxs = []
    names = []
    for i in range(len(dset_name)):
        if dset_name[i] == exp_name:
            src_idxs.append(i)
            names.append(dset_name[i])
    return torch.from_numpy(np.asarray(src_idxs)), names

def bbox_to_center_scale(bbox, dset_scale_factor=1.0, ref_bbox_size=200):
    if bbox is None:
        return None, None, None
    bbox = bbox.reshape(-1)
    bbox_size = dset_scale_factor * max(
        bbox[2] - bbox[0], bbox[3] - bbox[1])
    scale = bbox_size / ref_bbox_size
    center = np.stack(
        [(bbox[0] + bbox[2]) * 0.5,
         (bbox[1] + bbox[3]) * 0.5]).astype(np.float32)
    return center, scale, bbox_size
def bbox_area(bbox):
    if torch.is_tensor(bbox):
        if bbox is None:
            return 0.0
        xmin, ymin, xmax, ymax = torch.split(bbox.reshape(-1, 4), 1, dim=1)
        return torch.abs((xmax - xmin) * (ymax - ymin)).squeeze(dim=-1)
    else:
        if bbox is None:
            return 0.0
        xmin, ymin, xmax, ymax = np.split(bbox.reshape(-1, 4), 4, axis=1)
        return np.abs((xmax - xmin) * (ymax - ymin))
def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1
def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t
def keyps_to_bbox(keypoints, conf, reg = 0, img_size=None, clip_to_img=False,
                  min_valid_keypoints=6, scale=1.0):
    valid_keypoints = keypoints[conf > 0]
    if len(valid_keypoints) < min_valid_keypoints:
        return None
    H, W = img_size
    xmin, ymin = np.amin(valid_keypoints, axis=0)
    xmax, ymax = np.amax(valid_keypoints, axis=0)

    #We extend current min and max with cosntant value to cover blur region
    if reg > 0:
        xmin = xmin - reg
        ymin = ymin - reg
        xmax = xmax + reg
        ymax = ymax + reg

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin =0
        if xmax > W:
            xmax = W
        if ymax > H:
            ymax = H
    # Clip to the image
    if img_size is not None and clip_to_img:
        H, W, _ = img_size
        xmin = np.clip(xmin, 0, W)
        xmax = np.clip(xmax, 0, W)
        ymin = np.clip(ymin, 0, H)
        ymax = np.clip(ymax, 0, H)

    width = (xmax - xmin) * scale
    height = (ymax - ymin) * scale

    x_center = 0.5 * (xmax + xmin)
    y_center = 0.5 * (ymax + ymin)
    xmin = x_center - 0.5 * width
    xmax = x_center + 0.5 * width
    ymin = y_center - 0.5 * height
    ymax = y_center + 0.5 * height

    bbox = np.stack([xmin, ymin, xmax, ymax], axis=0).astype(np.float32)
    if bbox_area(bbox) > 0:
        return bbox
    else:
        return None

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa

def crop(img, center, scale, res, rot=0, dtype=np.float32):
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1],
                            center, scale, res, invert=1)) - 1
    # size of cropped image
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_shape = list(map(int, new_shape))
    new_img = np.zeros(new_shape, dtype=img.dtype)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]

    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    # Range to sample from original image
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]
            ] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    #  pixel_scale = 1.0 if new_img.max() > 1.0 else 255
    #  resample = pil_img.BILINEAR
    if not rot == 0:
        new_H, new_W, _ = new_img.shape

        rotn_center = (new_W / 2.0, new_H / 2.0)
        M = cv2.getRotationMatrix2D(rotn_center, rot, 1.0).astype(np.float32)

        new_img = cv2.warpAffine(new_img, M, tuple(new_shape[:2]),
                                 cv2.INTER_LINEAR_EXACT)
        new_img = new_img[pad:new_H - pad, pad:new_W - pad]

    output = cv2.resize(new_img, tuple(res), interpolation=cv2.INTER_LINEAR)
    return output.astype(np.float32)

def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)
    
def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat