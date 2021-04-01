import os
#comment this for making this run
#os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2
class Visualization_render(object):
    def __init__(self, focal_length, img_size, faces, root_dir ='visualize'):
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.deb_dir = os.path.join(root_dir, 'deblur')
        if not os.path.exists(self.deb_dir):
            os.makedirs(self.deb_dir)
        self.overlay_dir = os.path.join(root_dir, 'render')
        if not os.path.exists(self.overlay_dir):
            os.makedirs(self.overlay_dir)
        self.renderer = Renderer(focal_length=focal_length, img_res=img_size, faces=faces)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    def save_deblur(self, blur_imgs, sharp_imgs, deb_imgs, step):
        batch = blur_imgs.shape[0]
        de_bl = self.denorm_deb(blur_imgs, 'save_deblur')
        de_de = self.denorm_deb(deb_imgs, 'save_deblur')
        de_sh = self.denorm_deb(sharp_imgs, 'save_deblur')
        for i in range(batch):
            svd = os.path.join(self.deb_dir, '{}_{}.png'.format(step, i))
            stack = np.hstack([de_bl[i], de_de[i], de_sh[i]])
            cv2.imwrite(svd, cv2.cvtColor(stack, cv2.COLOR_BGR2RGB))
    def save_overlay(self, cam, pred_verts, gt_verts, imgs, epoch):
        new_imgs = self.denorm(imgs, 'overlay')
        self.renderer.vis_overlay_body_mesh(pred_verts, gt_verts, cam, new_imgs, self.overlay_dir, epoch)
    def get_dset_idxs(self, dset_name, batch_size, exp_name):
        idxs = []
        for i in range(batch_size):
            if dset_name[i] == exp_name:
                idxs.append(i)
        return torch.from_numpy(np.asarray(idxs))
    def denorm(self, img, save_type= 'save_deblur', imtype=np.uint8):
        images = img * torch.tensor(self.std , device=img.device).reshape(1,3,1,1)
        images = images + torch.tensor(self.mean, device=img.device).reshape(1,3,1,1)
        if save_type =='save_deblur':
            image_numpy = images.cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0
            return image_numpy.astype(imtype)
        return images
    def denorm_deb(self, img, save_type= 'save_deblur', imtype=np.uint8):
        images = img * torch.tensor([0.5, 0.5, 0.5] , device=img.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.5, 0.5, 0.5], device=img.device).reshape(1,3,1,1)
        if save_type =='save_deblur':
            image_numpy = images.cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0
            return image_numpy.astype(imtype)
        return images 
    # def save_deblur(self, blur, deblur, sharp, step):
    #     total = 10
    #     for i in range(total):
    #         svd = os.path.join(self.deb_dir, '{}_{}.png'.format(step, i))
    #         stack = np.hstack([blur[i], deblur[i], sharp[i]])
    #         cv2.imwrite(svd, cv2.cvtColor(stack, cv2.COLOR_BGR2RGB))
    #     pass
    def do_visualization(self, blur, deblur, sharp, body_pred, body_gt, cam, name, step):
        de_bl = self.denorm_deb(blur, 'save_deblur')
        de_de = self.denorm_deb(deblur, 'save_deblur')
        de_sh = self.denorm_deb(sharp, 'save_deblur')
        if 'own' in name:
            idx = self.get_dset_idxs(name, de_bl.shape[0], 'own')
        de_bl = de_bl[idx]
        # #save deblur result
        self.save_deblur(de_bl, de_de, de_sh, step)
        overlay_bl = self.denorm(blur, 'overlay')
        
        self.renderer.visualize_with_gt(body_pred.vertices.detach(), body_gt.vertices.detach(), cam.detach(), overlay_bl, name='pred', savename= self.overlay_dir, step = step)

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    # def __init__(self, focal_length=5000, img_res=224, faces=None):
    #     self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
    #                                    viewport_height=img_res,
    #                                    point_size=1.0)
    #     self.focal_length = focal_length
    #     self.camera_center = [img_res // 2, img_res // 2]
    #     self.faces = faces

    def __init__(self, focal_length=5000, img_res=256, width = 256, height = 256, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                       viewport_height=height,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [width // 2, height // 2]
        self.faces = faces
        self.loop_number = 0
    def vis_overlay_body_mesh(self, pred_vert, gt_vert, cam, images, path, epoch):
        batch = pred_vert.shape[0]
        ver_pred = pred_vert.clone().cpu().numpy()
        ver_gt = gt_vert.clone().cpu().numpy()
        images_tn = images.cpu()
        images_np = np.transpose(images_tn.numpy(), (0,2,3,1))
        for i in range(batch):
            rim_pred = torch.from_numpy(np.transpose(self.__call__(ver_pred[i], cam[i], images_np[i], base_color= (0.7, 0.5, 0.4, 1.0)), (2,0,1))).float()
            rim_gt = torch.from_numpy(np.transpose(self.__call__(ver_gt[i], cam[i], images_np[i], base_color= (0.3, 0.5, 0.8, 1.0)), (2,0,1))).float()

            rim_pred = rim_pred.detach().cpu().numpy()
            rim_gt = rim_gt.detach().cpu().numpy()

            rim_pred *= 255
            rim_gt *= 255  
            rim_pred = np.transpose(rim_pred.astype(np.uint8), (1, 2, 0))
            rim_gt = np.transpose(rim_gt.astype(np.uint8), (1, 2, 0)) 
            stack = np.hstack([rim_pred, rim_gt])
            str_file_name = '{}/{}_{}_{}.png'.format(path, epoch, i, 'pred')
            cv2.imwrite(str_file_name, cv2.cvtColor(stack, cv2.COLOR_BGR2RGB))

    def visualize_with_gt(self, pred_vert, gt_vert, camera_translation, images, name='pred', savename= 'test', step = 0):
        ver_pred = pred_vert.clone().cpu().numpy()
        ver_gt = gt_vert.clone().cpu().numpy()
        images_tn = images.cpu()
        images_np = np.transpose(images_tn.numpy(), (0,2,3,1))
        loop = 20
        for i in range(loop):
            rim_pred = torch.from_numpy(np.transpose(self.__call__(ver_pred[i], camera_translation[i], images_np[i], base_color= (0.7, 0.5, 0.4, 1.0)), (2,0,1))).float()
            rim_gt = torch.from_numpy(np.transpose(self.__call__(ver_gt[i], camera_translation[i], images_np[i], base_color= (0.7, 0.5, 0.4, 1.0)), (2,0,1))).float()

            rim_pred = rim_pred.detach().cpu().numpy()
            rim_gt = rim_gt.detach().cpu().numpy()
            
            rim_pred *= 255
            rim_gt *= 255  
            rim_pred = np.transpose(rim_pred.astype(np.uint8), (1, 2, 0))
            rim_gt = np.transpose(rim_gt.astype(np.uint8), (1, 2, 0)) 
            stack = np.hstack([rim_pred, rim_gt])
            str_file_name = '{}/{}_{}_{}.png'.format(savename, step, i, 'pred')
            cv2.imwrite(str_file_name, cv2.cvtColor(stack, cv2.COLOR_BGR2RGB))
        pass

    def visualize_tb(self, vertices, camera_translation, images, name = 'gt', savename = 'test.png', step = 0):
        vertices = vertices.cpu().numpy()
        loop = 20
        camera_translation = camera_translation.cpu().numpy()
        # images_tn = images[0].cpu()
        images_tn = images.cpu()
        images_np = np.transpose(images_tn.numpy(), (0,2,3,1))

        # sh_imgs = images[1]
        # sh_images_np = np.transpose(sh_imgs.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(loop):
            #write on folder
            if name=='pred':
                rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], base_color= (0.7, 0.5, 0.4, 1.0)), (2,0,1))).float()
                # rend_sh = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], sh_images_np[i], base_color= (0.7, 0.5, 0.4, 1.0)), (2,0,1))).float()
                str_file_name = '{}/{}_{}_{}.png'.format(savename, step, i, 'pred')
            elif name=='test':
                rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], base_color= (0.4, 0.8, 0.5, 1.0)), (2,0,1))).float()
                str_file_name = 'visualize_test/{}_{}_{}.png'.format(self.loop_number, i, 'test')
            else:
                rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], base_color= (0.2, 0.4, 0.9, 1.0)), (2,0,1))).float()
                str_file_name = 'visualize_test/{}_{}_{}.png'.format(self.loop_number, i, 'gt')
            
            fold_img = rend_img.detach().cpu().numpy()
            # sh_fold_img = rend_sh.cpu().numpy()
            fold_img *= 255
            # sh_fold_img *= 255
            fold_img = np.transpose(fold_img.astype(np.uint8), (1, 2, 0))  
            # sh_fold_img = np.transpose(sh_fold_img.astype(np.uint8), (1, 2, 0)) 

            # stack = np.hstack([fold_img, sh_fold_img]) 
            # cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
            # cv2.imwrite(str_file_name, cv2.cvtColor(fold_img, cv2.COLOR_BGR2RGB))

            cv2.imwrite(str_file_name, cv2.cvtColor(fold_img, cv2.COLOR_BGR2RGB))
        self.loop_number+=1
        if name=='gt':
            self.loop_number+=1

    def __call__(self, vertices, camera_translation, image, cam_for_render = None, base_color = (0.8, 0.3, 0.3, 1.0)):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor= base_color)

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation

        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        if cam_for_render is not None:
            camera = pyrender.IntrinsicsCamera(fx=cam_for_render[0], fy=cam_for_render[0],
                                           cx=cam_for_render[1], cy=cam_for_render[2])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)
        
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        
        if cam_for_render is not None:
            output_img = (255 * color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img
