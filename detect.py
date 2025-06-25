import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO('./best.pt') # select your model.pt path
    model.predict(source='jpgs',
                  imgsz=640,
                  project='runs/ours',
                  name='exp',
                  save=True,
                  conf=0.5,
                  iou=0.5,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  show_conf=True, # do not show prediction confidence
                  show_labels=True, # do not show prediction labels
                  #save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )