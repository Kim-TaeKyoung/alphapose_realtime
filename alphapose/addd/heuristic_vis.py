import pandas as pd
from matpoltlib import pyplot as plt
import matplotlib.patches as patches
import cv2

def heuristic_visualize(df_pose, viewer_result, image):

    fig = plt.figure(figsize=(6.4*2, 4.8*2), dpi=200)
    ax = fig.add_subplot(111)
    width = image.size()[0]
    height = image.size()[1]
    ax.imshow(image)
    
    current_viewer = df_pose[df_pose['image_id'] == image_name]['idx'].values
    current_viewer_box = df_pose[df_pose['image_id'] == image_name]['box'].values
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

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img