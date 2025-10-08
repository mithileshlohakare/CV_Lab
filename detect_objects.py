import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def plot_results(img, predictions, threshold=0.5):
    """
    Plots the image with detection results.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    ax = plt.gca()

    # List of COCO dataset category names
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
        'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
        'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Extract boxes, labels, and scores from predictions
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Draw a bounding box for each detection that exceeds the confidence threshold
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            xmin, ymin, xmax, ymax = box
            # Create a rectangle patch
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            # Add the label and score
            ax.text(xmin, ymin, f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}",
                    bbox=dict(facecolor='yellow', alpha=0.5))
    
    # Display the plot
    plt.axis('off')
    plt.show()


# --- Main execution ---
if __name__ == "__main__":
    # IMPORTANT: Change this to the path of your image file
    image_path = "istockphoto-1252455620-612x612.jpg" 

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        print("Please make sure the image file is in the same directory as the script, or provide the full path.")
    else:
        # 1. Load the image
        img = Image.open(image_path).convert("RGB")

        # 2. Load a pre-trained Faster R-CNN model
        print("Loading pre-trained Faster R-CNN model...")
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()  # Set the model to evaluation mode
        print("Model loaded.")

        # 3. Convert the image to a tensor
        img_tensor = F.to_tensor(img)

        # 4. Perform inference
        print("Performing object detection...")
        with torch.no_grad():
            predictions = model([img_tensor])
        print("Detection complete.")

        # 5. Plot results on the image
        plot_results(img, predictions, threshold=0.7)