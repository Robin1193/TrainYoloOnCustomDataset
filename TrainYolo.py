from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="config.yaml", epochs=50)  # train the model

metrics = model.val()  # evaluate model performance on the validation set
results = model("test_dogs.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format