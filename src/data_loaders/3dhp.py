"""
Open up mpi_inf_3dhp.
TRAINING:
For each subject & sequence there is annot.mat
What is in annot.mat:
  'frames': number of frames, N
  'univ_annot3': (14,) for each camera of N x 84 -> Why is there univ for each camera if it's univ..?
  'annot3': (14,) for each camera of N x 84
  'annot2': (14,) for each camera of N x 56
  'cameras':
  In total there are 28 joints, but H3.6M subsets are used.
  The image frames are unpacked in:
  BASE_DIR/S%d/Seq%d/video_%d/frame_%06.jpg
TESTING:
  'valid_frame': N_frames x 1
  'annot2': N_frames x 1 x 17 x 2
  'annot3': N_frames x 1 x 17 x 3
  'univ_annot3': N_frames x 1 x 17 x 3
  'bb_crop': this is N_frames x 34 (not sure what this is..)
  'activity_annotation': N_frames x 1 (of integer indicating activity type
  The test images are already in jpg.

Folder Structure for data:
    MP1
        --->S1
            --->Seq1
                --->imageFrames
                    --->video_X (0,1,2,4,5,6,7,8)
                        --->frame_000000.jpg ... (All images)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from os.path import join, exists
# To go to h36m joints:
# training joints have 28 joints
# test joints are 17 (H3.6M subset in CPM order)

def sample_frames(gt3ds):
    use_these = np.zeros(gt3ds.shape[0], bool)
    # Always use_these first frame.
    use_these[0] = True
    prev_kp3d = gt3ds[0]
    for itr, kp3d in enumerate(gt3ds):
        if itr > 0:
            # Check if any joint moved more than 200mm.
            if not np.any(np.linalg.norm(prev_kp3d - kp3d, axis=1) >= 200):
                continue
        use_these[itr] = True
        prev_kp3d = kp3d

    return use_these

def get_paths(base_dir, sub_id, seq_id):
    data_dir = join(base_dir, 'S%d' % sub_id, 'Seq%d' % seq_id)
    anno_path = join(data_dir, 'annot.mat')
    img_dir = join(data_dir, 'imageFrames')
    return img_dir, anno_path


def read_mat(path):
    from scipy.io import loadmat
    res = loadmat(path, struct_as_record=True, squeeze_me=True)

    cameras = res['cameras']
    annot2 = np.stack(res['annot2'])
    annot3 = np.stack(res['annot3'])
    frames = res['frames']

    # univ_annot3 = np.stack(res['univ_annot3'])

    return frames, cameras, annot2, annot3


def mpi_inf_3dhp_to_lsp_idx():
    # For training, this joint_idx gives names 17
    raw_to_h36m17_idx = np.array(
        [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]) - 1
    names_17 = [
        'Head', 'Neck', 'R Shoulder', 'R Elbow', 'R Wrist', 'L Shoulder',
        'L Elbow', 'L Wrist', 'R Hip', 'R Knee', 'R Ankle', 'L Hip', 'L Knee',
        'L Ankle', 'Pelvis', 'Spine', 'Head'
    ]
    want_names = [
        'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
        'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck',
        'Head'
    ]

    h36m17_to_lsp_idx = [names_17.index(j) for j in want_names]

    raw_to_lsp_idx = raw_to_h36m17_idx[h36m17_to_lsp_idx]

    return raw_to_lsp_idx, h36m17_to_lsp_idx

joint_idx2lsp, test_idx2lsp = mpi_inf_3dhp_to_lsp_idx()
def read_camera(base_dir):
    cam_path = join(base_dir, 'S1/Seq1/camera.calibration')
    lines = []
    with open(cam_path, 'r') as f:
        for line in f:
            content = [x for x in line.strip().split(' ') if x]
            lines.append(content)

    def get_cam_info(block):
        cam_id = int(block[0][1])
        # Intrinsic
        intrinsic = block[4][1:]
        K = np.array([np.float(cont) for cont in intrinsic]).reshape(4, 4)
        # Extrinsic:
        extrinsic = block[5][1:]
        Ext = np.array([float(cont) for cont in extrinsic]).reshape(4, 4)
        return cam_id, K, Ext

    # Skip header
    lines = lines[1:]
    # each camera is 7 lines long.
    num_cams = int(len(lines) / 7)
    cams = {}
    for i in range(num_cams):
        cam_id, K, Ext = get_cam_info(lines[7 * i:7 * i + 7])
        cams[cam_id] = K

    return cams

def get_all_data(base_dir, sub_id, seq_id, cam_ids, all_cam_info):
    img_dir, anno_path = get_paths(base_dir, sub_id, seq_id)
    # Get data for all cameras.
    frames, _, annot2, annot3 = read_mat(anno_path)

    all_gt2ds, all_gt3ds, all_img_paths = [], [], []
    all_cams = []
    for cam_id in cam_ids:
        base_path = join(img_dir, 'video_%d' % cam_id, 'frame_%06d.jpg')
        num_frames = annot2[cam_id].shape[0]
        gt2ds = annot2[cam_id].reshape(num_frames, -1, 2)
        gt3ds = annot3[cam_id].reshape(num_frames, -1, 3)
        # Convert N x 28 x . to N x 14 x 2, N x 14 x 3
        gt2ds = gt2ds[:, joint_idx2lsp, :]
        gt3ds = gt3ds[:, joint_idx2lsp, :]
        img_paths = [base_path % (frame + 1) for frame in frames]
        if gt3ds.shape[0] != len(img_paths):
            print('Not same paths?')
            import ipdb
            ipdb.set_trace()
        use_these = sample_frames(gt3ds)
        all_gt2ds.append(gt2ds[use_these])
        all_gt3ds.append(gt3ds[use_these])
        K = all_cam_info[cam_id]
        flength = 0.5 * (K[0, 0] + K[1, 1])
        ppt = K[:2, 2]
        flengths = np.tile(flength, (np.sum(use_these), 1))
        ppts = np.tile(ppt, (np.sum(use_these), 1))
        cams = np.hstack((flengths, ppts))
        all_cams.append(cams)
        all_img_paths += np.array(img_paths)[use_these].tolist()

    all_gt2ds = np.vstack(all_gt2ds)
    all_gt3ds = np.vstack(all_gt3ds)
    all_cams = np.vstack(all_cams)

    return all_img_paths, all_gt2ds, all_gt3ds, all_cams

def process_mpi_inf_3dhp_train(data_dir, out_dir, is_train=False):
    if is_train:
        out_dir = join(out_dir, 'train')
        print('!train set!')
        sub_ids = range(1, 8)  # No S8!
        seq_ids = range(1, 3)
        cam_ids = [0, 1, 2, 4, 5, 6, 7, 8]
    else:  # Full set!!
        out_dir = join(out_dir, 'trainval')
        print('doing the full train-val set!')
        sub_ids = range(1, 9)
        seq_ids = range(1, 3)
        cam_ids = [0, 1, 2, 4, 5, 6, 7, 8]

    # if not exists(out_dir):
    #     makedirs(out_dir)

    # Load all data & shuffle it,,
    all_gt2ds, all_gt3ds, all_img_paths = [], [], []
    all_cams = []
    all_cam_info = read_camera(data_dir)

    for sub_id in sub_ids:
        for seq_id in seq_ids:
            print('collecting S%d, Seq%d' % (sub_id, seq_id))
            # Collect all data for each camera.
            # img_paths: N list
            # gt2ds/gt3ds: N x 17 x 2, N x 17 x 3
            img_paths, gt2ds, gt3ds, cams = get_all_data(
                data_dir, sub_id, seq_id, cam_ids, all_cam_info)

            all_img_paths += img_paths
            all_gt2ds.append(gt2ds)
            all_gt3ds.append(gt3ds)
            all_cams.append(cams)

    all_gt2ds = np.vstack(all_gt2ds)
    all_gt3ds = np.vstack(all_gt3ds)
    all_cams = np.vstack(all_cams)
    assert (all_gt3ds.shape[0] == len(all_img_paths))
    # Now shuffle it all.
    shuffle_id = np.random.permutation(len(all_img_paths))
    all_img_paths = np.array(all_img_paths)[shuffle_id]
    all_gt2ds = all_gt2ds[shuffle_id]
    all_gt3ds = all_gt3ds[shuffle_id]
    all_cams = all_cams[shuffle_id]
    import h5py
    h5file = h5py.File('annot.h5', 'w')
    h5file.create_dataset('gt2d', data = all_gt2ds)
    h5file.create_dataset('gt3d', data = all_gt3ds)
    dt = h5py.string_dtype(encoding='ascii')
    all_img_paths = all_img_paths.astype(h5py.string_dtype(encoding='ascii'))
    h5file.create_dataset('imagename', data = all_img_paths, dtype=dt)
    h5file.close()

if __name__ == '__main__':
    # The location of the folder where the folders S1 - S8.. are present
    # The annotation file will be saved to this location
    process_mpi_inf_3dhp_train('../../../../data/MP1/mpi_inf_3dhp/', '/' )
