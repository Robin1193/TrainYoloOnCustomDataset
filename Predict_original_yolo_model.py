from ultralytics import YOLO

print("Start predicting test dogs with original yolo model")

# Load a model
model = YOLO("yolov8m.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["test_dogs.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result_test_dogs_original_yolo.jpg")  # save to disk