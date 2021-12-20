import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
import pandas as pd

THRESHOLD_DICT = {'confidence_threshold': 0.6, 
                  'score_threshold': 1.0,
                  'height_ratio_threshold': 1./16, 
                  'lowest_point_ratio_threhold': 2./5,
                  'face_threshold': [0.5, 0.5, 0.5],
                  'max_knee_angle_threshold': 40,
                 'eye_dist_ratio_threshold': 0.10}

def angle(a, b, c):
    ba = a - b
    bc = c - b
    return  np.rad2deg(np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))))

def consecutive(data, stepsize=0):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def smoothing(d, min_keeping_frame):
    d_cons = consecutive(d)
    before = d_cons[0][0]
    for i in range(1, len(d_cons)):
        if d_cons[i][0] != before:
            if (len(d_cons[i]) < min_keeping_frame):
                d_cons[i] += 1
                d_cons[i] %= 2
        before = d_cons[i][0]
    return np.concatenate(d_cons)

def df_hit_miss(df_pose: pd.DataFrame, index: float):
  hit = []
  for i, idx in df_pose['idx'].iteritems():
    if index in idx:
      hit.append(df_pose.iloc[i])
  
  return pd.DataFrame(hit, columns=df_pose.columns)

def pose_analysis(pose: dict, image_size = (1280, 720), confidence_threshold=0.6, score_threshold=1.0, height_ratio_threshold=1./16, lowest_point_ratio_threhold=2./5, face_threshold=[0.5, 0.5, 0.5], max_knee_angle_threshold=30, eye_dist_ratio_threshold=0.2, is_valid=0, is_front=0, is_watching=0, is_sitting=0):
    keypoints = np.array(pose['keypoints']).reshape(-1, 3)
    height = (np.max(keypoints[:, 1])-np.min(keypoints[:, 1])) # Not real height, more like ankle-to--eye length
    height_ratio = height / image_size[1]
    lowest_point = np.max(keypoints[:, 1])  # (0, 0) starts from upper left corner, thus max y points is the lowest point
    is_valid = 1 if (np.mean(keypoints[:, 2]) > confidence_threshold) and (pose['score'] > score_threshold) and (height_ratio > height_ratio_threshold) else 0
    if is_valid:
        # checking heading direction (front/back)
        Nose, LEye, REye, LEar, REar, Lshoulder, Rshoulder = keypoints[0], keypoints[1], keypoints[2], keypoints[3], keypoints[4], keypoints[5], keypoints[6]
        LElbow, RElbow, LWrist, RWrist, LHip, RHip = keypoints[7], keypoints[8], keypoints[9], keypoints[10], keypoints[11], keypoints[12]
        LKnee, RKnee, LAnkle, RAnkle = keypoints[13], keypoints[14], keypoints[15], keypoints[16]
        torso_center = np.mean([Lshoulder, Rshoulder], axis=0)  
        lower_center = np.mean([LHip, RHip], axis=0)
        body_length = (np.sum(torso_center[:2] - lower_center[:2]) ** 2) ** 0.5
        eye_dist = (LEye[0]-REye[0])
        right_knee_angle = 180-angle(torso_center, lower_center, RKnee)
        left_knee_angle = 180-angle(torso_center, lower_center, LKnee)
        knee_parity = np.sign(right_knee_angle * left_knee_angle)
        max_knee_angle = np.max([np.abs(right_knee_angle), np.abs(left_knee_angle)])
        
        is_sitting = (max_knee_angle > max_knee_angle_threshold) and (knee_parity)
        is_front = 1 if (Nose[2] >= face_threshold[0]) and (LEye[2] >= face_threshold[1]) and (REye[2] >= face_threshold[2]) and eye_dist > 0 else 0  # nose confidence
        
        if is_front:
            # checking body position (left/right w.r.t. image center)
            torso_center = np.mean([Lshoulder, Rshoulder], axis=0)  
            image_center = (image_size[0]/2, image_size[1]/2)
            pos_diff= (torso_center[0] - image_center[0]) / (image_size[0] * 0.5)  # positive : right, negative
            nose_torso_diff = (Nose[0] - torso_center[0])  # positive : right, negative
            #nose_torso_diff_ratio_threshold = pos_diff * max_nose_torso_diff_ratio_threshold
            eye_dist_ratio = eye_dist/body_length
            #is_watching = 1 if (pos_diff * nose_torso_diff <= 0) and (eye_dist_ratio > eye_dist_ratio_threshold) else 0
            is_watching = 1 if (eye_dist_ratio > eye_dist_ratio_threshold) else 0
    return is_valid, is_front, is_watching, is_sitting

def viewer_detection(df_pose, image_size, fps, min_keeping_frame=0.3, threshold_dict=THRESHOLD_DICT, return_invalid=False):
    '''
    image_results : algorithm output for each frame, -1 -> invalid frame, 0 -> valid but not watching, 1 -> valid and watching
    '''
    spf = 1 / fps  # Second Per Frame
    ret = []

    # Unpacking every sub-lists, 
    # due to the fact that we only detect pedestrian.

    for i, idx in df_pose['idx'].iteritems():
      max_idx = 0
      if len(idx) > 1:
        idxs = [sub_idx[0] for sub_idx in idx]
        max_idx = max(max_idx, max(idxs))
        df_pose.at[i, 'idx'] = idxs
      else:
        max_idx = max(max_idx, idx[0])

    for idx in range(1, int(max_idx)+1):
        template = {'idx': idx,
                    'valid_idx':False,
                    'image_id_list':None,
                    'image_results':None,
                    'T_total':0,
                    'T_valid':0.,
                    'T_front':0.,
                    'T_attention':0.,
                    'T_sit':0.}
        
        data = df_hit_miss(df_pose, idx)
        if not data.empty:
          result = np.array(list(data.apply(pose_analysis, axis=1, image_size=image_size, **threshold_dict)))

          ## smoothing
          result[:, 1] = smoothing(result[:, 1], min_keeping_frame) * result[:, 0]
          result[:, 2] = smoothing(result[:, 2], min_keeping_frame) * result[:, 1]
          result[:, 3] = smoothing(result[:, 3], min_keeping_frame) * result[:, 0]
          
          F_total = len(result)
          F_valid = np.sum(result[:, 0])
          F_front = np.sum(result[:, 1])
          F_attention = np.sum(result[:, 2])
          F_sit = np.sum(result[:, 3])

          template['valid_idx'] = True
          template['image_id_list'] = list(data['image_id'].values)
          template['image_results'] = result
          template['T_total'] = F_total * spf
          template['T_valid'] = F_valid * spf
          template['T_front'] = F_front * spf
          template['T_attention'] = F_attention * spf
          template['T_sit'] = F_sit * spf
          ret.append(template)

    df_ret = pd.DataFrame(ret)
    if return_invalid:
        return df_ret
    else:
        return df_ret[df_ret['valid_idx']]

def heuristic_visualize(df_pose, viewer_result, image):
    width = image.shape[1]
    height = image.shape[0]
    fig = plt.figure(figsize=((width / 100) * 2, (height / 100) * 2), dpi=200)
    ax = fig.add_subplot(111)
    ax.margins(0)
    ax.imshow(image)

    for i, idx in enumerate(current_viewer):
        d = viewer_result[viewer_result['idx']==idx]
        arg = np.argwhere(np.array(d['image_id_list'].item()) == f'{image_name}').item()
        stat = d['image_results'].item()[arg]
        bbox = current_viewer_box[i]
        rect1 = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((bbox[0] + min(width - (bbox[0]+max(bbox[2], 35)), 0), (bbox[1]-55)), max(bbox[2], 35), 50, linewidth=1, edgecolor='r', facecolor='white', alpha=0.7)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        rx, ry = rect2.get_xy()
        cx = rx + rect2.get_width()/2.0
        cy = ry + rect2.get_height()/2.0
        stat = ['O' if x == 1 else 'X' for x in stat]
        ax.annotate(f'idx:{idx}\nvalid:{stat[0]}\nfront:{stat[1]}\nwatch:{stat[2]}\nsit:{stat[3]}', (cx, cy), color='k', fontsize=8, ha='center', va='center')
    plt.axis('off')

    fig.canvas.draw()
    new_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    new_image = new_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return new_image