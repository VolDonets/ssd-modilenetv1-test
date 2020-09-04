### PREPROCESSING

import numpy as np
from PIL import Image, ImageDraw, ImageColor
import math
import matplotlib.pyplot as plt

# open and display image file
img = Image.open("src/images/dog.jpg")
plt.axis('off')
plt.imshow(img)
plt.show()

# reshape the flat array returned by img.getdata() to HWC and than add an additial    
# dimension to make NHWC, aka a batch of images with 1 image in it
img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)





### INFERENCE

import onnxruntime as rt

# produce outputs in this order
outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

# load model and run inference
sess = rt.InferenceSession("src/ssd_mobilenet_v1_10.onnx")
result = sess.run(outputs, {"image_tensor:0": img_data})
num_detections, detection_boxes, detection_scores, detection_classes = result

# print number of detections
print(num_detections)
print(detection_classes)







### POSTPROCESSING

coco_classes = [line.rstrip('\n') for line in open('src/coco_classes.txt')]

# draw boundary boxes and label for each detection
def draw_detection(draw, d, c):
    width, height = draw.im.size
    # the box is relative to the image size so we multiply with height and width to get pixels
    top = int(d[0] * height)
    left = int(d[1] * width)
    bottom = int(d[2] * height)
    right = int(d[3] * width)
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    label = coco_classes[c]
    label_size = draw.textsize(label)
    if top - label_size[1] >= 0:
        text_origin = tuple(np.array([left, top - label_size[1]]))
    else:
        text_origin = tuple(np.array([left, top + 1]))
    color = ImageColor.getrgb("red")
    thickness = 0
    draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], 
    outline=color)
    draw.text(text_origin, label, fill=color)
    print(text_origin)
    print(label)
    
# loop over the results - each returned tensor is a batch
batch_size = num_detections.shape[0]
draw = ImageDraw.Draw(img)
for batch in range(0, batch_size):
    for detection in range(0, int(num_detections[batch])):
        c = int(detection_classes[batch][detection])
        d = detection_boxes[batch][detection]
        draw_detection(draw, d, c)

# show image file with object detection boundary boxes and labels 
plt.figure(figsize=(80, 40))
plt.axis('off')
plt.imshow(img)
plt.show()