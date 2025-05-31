from data_loader import train_loader, val_loader, test_loader
from model_resnet import build_model
from train_model import train
from evaluate_model import evaluate_on_test
from visualize_gradcam import apply_gradcam

class_names = train_loader.dataset.classes
num_classes = len(class_names)

model, device = build_model(num_classes)

train(model, train_loader, val_loader, device, num_epochs=10)

evaluate_on_test(model, test_loader, class_names, device)

image_path_flora = "Dataset_split/test/Flora/flora_018.png"  
apply_gradcam(model, image_path_flora, class_names, device)

image_path_geo = "Dataset_split/test/Geo/geo_021.png"  
apply_gradcam(model, image_path_geo, class_names, device)

image_path_zoo = "Dataset_split/test/Zoo/zoo_036.png"  
apply_gradcam(model, image_path_zoo, class_names, device)