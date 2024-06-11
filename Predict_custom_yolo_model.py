from ultralytics import YOLO

print("Start predicting test dogs with custom yolo model")

# TODO: Anpassen Model
# Load a model
model = YOLO("runs/detect/train49/weights/best.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model("test_dogs.jpg")  # return a list of Results objects


# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result_test_dogs_custom_yolo.jpg")  # save to disk