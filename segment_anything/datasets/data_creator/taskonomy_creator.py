
import torch
import numpy as np
import pickle

import math 
import json
from PIL import Image

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def compute_projection_matrix(
    znear, zfar, fov, aspect_ratio, degrees: bool
) -> torch.Tensor:
    """
    Compute the calibration matrix K of shape (N, 4, 4)

    Args:
        znear: near clipping plane of the view frustrum.
        zfar: far clipping plane of the view frustrum.
        fov: field of view angle of the camera.
        aspect_ratio: aspect ratio of the image pixels.
            1.0 indicates square pixels.
        degrees: bool, set to True if fov is specified in degrees.

    Returns:
        torch.FloatTensor of the calibration matrix with shape (N, 4, 4)
    """
    K = torch.zeros((1, 4, 4), dtype=torch.float32)
    ones = torch.ones((1), dtype=torch.float32)
    if degrees:
        fov = (np.pi / 180) * fov

    if not torch.is_tensor(fov):
        fov = torch.tensor(fov)
    tanHalfFov = torch.tan((fov / 2))
    max_y = tanHalfFov * znear
    min_y = -max_y
    max_x = max_y * aspect_ratio
    min_x = -max_x

    # NOTE: In OpenGL the projection matrix changes the handedness of the
    # coordinate frame. i.e the NDC space positive z direction is the
    # camera space negative z direction. This is because the sign of the z
    # in the projection matrix is set to -1.0.
    # In pytorch3d we maintain a right handed coordinate system throughout
    # so the so the z sign is 1.0.
    z_sign = 1.0

    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    K[:, 0, 0] = 2.0 * znear / (max_x - min_x)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    K[:, 1, 1] = 2.0 * znear / (max_y - min_y)
    K[:, 0, 2] = (max_x + min_x) / (max_x - min_x)
    K[:, 1, 2] = (max_y + min_y) / (max_y - min_y)
    K[:, 3, 2] = z_sign * ones

    # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
    # is at the near clipping plane and z = 1 when the point is at the far
    # clipping plane.
    K[:, 2, 2] = z_sign * zfar / (zfar - znear)
    K[:, 2, 3] = -(zfar * znear) / (zfar - znear)

    return K

def _get_cam_to_world_R_T_K(point_info, device='cpu'):
    EULER_X_OFFSET_RADS = math.radians(90.0)
    location = point_info['camera_location']
    rotation = point_info['camera_rotation_final']
    fov      = point_info['field_of_view_rads']

    
    # Recover cam -> world
    ex, ey, ez = rotation
    R = euler_angles_to_matrix(torch.tensor(
                [(ex - EULER_X_OFFSET_RADS, -ey, -ez)],
                dtype=torch.double, device=device), 'XZY')
    Tx, Ty, Tz = location
    T = torch.tensor([[-Tx, Tz, Ty]], dtype=torch.double, device=device) 



    # P3D expects world -> cam
    R_inv = R.transpose(1,2)
    T_inv = -R.bmm(T.unsqueeze(-1)).squeeze(-1)
    # T_inv = -R.bmm(T.unsqueeze(-1)).squeeze(-1)
    # T_inv = T
    # R_inv = R 
    K = compute_projection_matrix(znear=0.001, zfar=512.0, fov=fov, aspect_ratio=1.0, degrees=False)
    
    return dict(
        cam_to_world_R=R_inv.squeeze(0).float(),
        cam_to_world_T=T_inv.squeeze(0).float(),
        proj_K=K.squeeze(0).float(),
        proj_K_inv=K[:,:3,:3].inverse().squeeze(0).float())

taskonomy_pkl_dict = []

file_name_image = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/taskonomy/rgb/ackermanville/rgb/point_0_view_0_domain_rgb.png'

with open('/cpfs01/user/zhanghanxue/segment-anything/point_info/point_0_view_0_domain_point_info.json', 'rb') as f:
    point_info = json.load(f)
import ipdb;ipdb.set_trace()
P = _get_cam_to_world_R_T_K(point_info)['proj_K'].numpy()
w = h = 512 # omnidata should all be 512
fx, fy, cx, cy = P[0,0]*w/2, P[1,1]*h/2, (w-P[0,2]*w)/2, (P[1,2]*h+h)/2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).reshape(1, 3, 3)

depth = np.array(Image.open('/cpfs01/shared/opendrivelab/opendrivelab_hdd/taskonomy/depth/ackermanville/depth_zbuffer/point_0_view_0_domain_depth_zbuffer.png')).astype(np.float32) / 512.
name = 'ackermanville'
np.save('/cpfs01/shared/opendrivelab/opendrivelab_hdd/taskonomy/npy_depth/' + name + '_npy_depth.npy', depth)
import ipdb;ipdb.set_trace()
print(depth)

tmp_dict={'K': K,
    'img_path': file_name_image,
    'depth_path': '/cpfs01/shared/opendrivelab/opendrivelab_hdd/taskonomy/npy_depth/' + name + '_npy_depth.npy'}

taskonomy_pkl_dict.append(tmp_dict)

with open('/cpfs01/shared/opendrivelab/opendrivelab_hdd/taskonomy/test0.pkl', 'wb') as file:
    pickle.dump(taskonomy_pkl_dict, file)