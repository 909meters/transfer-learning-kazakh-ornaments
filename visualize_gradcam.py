import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

def apply_gradcam(model, image_path, class_names, device):
    # Подготовка изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True  # ✅ ВКЛЮЧАЕМ градиенты

    # Слои и буферы
    gradients = []
    activations = []

    # Выбираем последний сверточный слой
    target_layer = model.layer4[1].conv2

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Регистрируем хуки
    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_full_backward_hook(backward_hook)

    # Прямой проход
    model.train()  # 🔥 НЕ eval(), чтобы градиенты работали
    output = model(input_tensor)
    pred_class = output.argmax().item()
    print("Predicted:", class_names[pred_class])

    # Обратный проход
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    # Удаляем хуки
    hook_forward.remove()
    hook_backward.remove()

    # Проверка: градиенты и активации
    if not gradients or not activations:
        print("❌ No gradients or activations found.")
        return

    # Обработка
    grad = gradients[0].detach().cpu().numpy()
    act = activations[0].detach().cpu().numpy()
    weights = np.mean(grad, axis=(2, 3))[0]

    cam = np.zeros(act.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()

    # Наложение
    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    result = heatmap * 0.4 + img_np * 0.6

    # Визуализация
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM")
    plt.imshow(cam, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(result.astype(np.uint8))
    plt.axis('off')

    plt.tight_layout()
    plt.show()
