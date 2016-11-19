from PIL import Image, ImageDraw
from LSPGlobals import FLAGS
from LSPGlobals import BodyParts as p
from tensorflow.python.platform import gfile
import os.path as pt
import numpy as np

labels_ = 0
drawer_ = 0


def g(body_part_id):
    global labels_
    return labels_[body_part_id, 0], labels_[body_part_id, 1]


def draw(start_part, stop_part, fill="red"):
    drawer_.line([g(start_part), g(stop_part)], width=3, fill=fill)


def draw_pose_on_image(fim, arg_labels, folder, fileid, file_name=None, reshaped=False):
    global labels_
    global drawer_
    
    if not reshaped:
        labels_ = arg_labels.reshape([FLAGS.label_count, FLAGS.label_size])[:,:2]
    else:
        labels_ = arg_labels

    image_array = ((fim + 0.5)*255)
    im = Image.fromarray(image_array.astype(np.uint8))
    drawer_ = ImageDraw.Draw(im)
    
    # middle parts
    draw(p.Head_top, p.Neck)
    draw(p.Left_hip, p.Right_hip)
    
    # left arm
    draw(p.Neck, p.Left_shoulder)
    draw(p.Left_shoulder, p.Left_elbow)
    draw(p.Left_elbow, p.Left_wrist)
    
    # left leg
    draw(p.Left_shoulder, p.Left_hip)
    draw(p.Left_hip, p.Left_knee)
    draw(p.Left_knee, p.Left_ankle)
    
    # right arm
    draw(p.Neck, p.Right_shoulder)
    draw(p.Right_shoulder, p.Right_elbow)
    draw(p.Right_elbow, p.Right_wrist)
    
    # right leg
    draw(p.Right_shoulder, p.Right_hip)
    draw(p.Right_hip, p.Right_knee)
    draw(p.Right_knee, p.Right_ankle)

    if not gfile.Exists(folder):
        gfile.MakeDirs(folder)
        
    if file_name is not None:
        im.save(pt.join(folder, file_name))
    else:
        im.save(pt.join(folder, "%05d.jpg" % fileid))
        
    return
