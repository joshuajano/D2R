import torch
from torch.nn import functional as F

class ContinuousRotReprDecoder(torch.nn.Module):
    ''' Decoder for transforming a latent representation to rotation matrices
        Implements the decoding method described in:
        "On the Continuity of Rotation Representations in Neural Networks"
    '''
    def __init__(self, num_angles, dtype=torch.float32, mean=None,
                 **kwargs):
        super(ContinuousRotReprDecoder, self).__init__()
        self.num_angles = num_angles
        self.dtype = dtype

        if isinstance(mean, dict):
            mean = mean.get('cont_rot_repr', None)
        if mean is None:
            mean = torch.tensor(
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                dtype=self.dtype).unsqueeze(dim=0).expand(
                    self.num_angles, -1).contiguous().view(-1)

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        mean = mean.reshape(-1, 6)

        if mean.shape[0] < self.num_angles:
            logger.debug(mean.shape)
            mean = mean.repeat(
                self.num_angles // mean.shape[0] + 1, 1).contiguous()
            mean = mean[:self.num_angles]
        elif mean.shape[0] > self.num_angles:
            mean = mean[:self.num_angles]

        mean = mean.reshape(-1)
        self.register_buffer('mean', mean)

    def get_type(self):
        return 'cont_rot_repr'

    def extra_repr(self):
        msg = 'Num angles: {}\n'.format(self.num_angles)
        msg += 'Mean: {}'.format(self.mean.shape)
        return msg

    def get_param_dim(self):
        return 6

    def get_dim_size(self):
        return self.num_angles * 6

    def get_mean(self):
        return self.mean.clone()

    def to_offsets(self, x):
        latent = x.reshape(-1, 3, 3)[:, :3, :2].reshape(-1, 6)
        return (latent - self.mean).reshape(x.shape[0], -1, 6)

    def encode(self, x, subtract_mean=False):
        orig_shape = x.shape
        if subtract_mean:
            raise NotImplementedError
        output = x.reshape(-1, 3, 3)[:, :3, :2].contiguous()
        return output.reshape(
            orig_shape[0], orig_shape[1], 3, 2)

    def forward(self, module_input):
        batch_size = module_input.shape[0]
        reshaped_input = module_input.view(-1, 3, 2)

        # Normalize the first vector
        b1 = F.normalize(reshaped_input[:, :, 0].clone(), dim=1)

        dot_prod = torch.sum(
            b1 * reshaped_input[:, :, 1].clone(), dim=1, keepdim=True)
        # Compute the second vector by finding the orthogonal complement to it
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=1)
        # Finish building the basis by taking the cross product
        b3 = torch.cross(b1, b2, dim=1)
        rot_mats = torch.stack([b1, b2, b3], dim=-1)

        return rot_mats.view(batch_size, -1, 3, 3)