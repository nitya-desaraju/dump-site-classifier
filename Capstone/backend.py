import os
import io
import base64
import pickle
import numpy as np
import PIL.Image
import torch
import torchvision
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from matplotlib import cm
from torchvision.models import resnet


# Assuming your 'utils' folder with 'image_processor.py' is in the same directory
from utils.image_processor import ImageProcessor

# --- Model Configuration ---
CATS = ["suspicious_site"]
STATE_DICT_PATH = "weights/checkpoint.pth"
CLASSIFIER_MODEL_PATH = "dump_classifier_full.pth"
model_arch = 'architecture.resnet50_fpn'
# Forcing CPU mode to resolve the CUDA error on your MacBook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global variable to hold the loaded classification model
classifier_model = None

# --- Core Model Functions ---

def generate_heatmap(img_path):
  """Generates a heatmap using the first model to find potential dump sites."""
  ip = ImageProcessor(CATS, STATE_DICT_PATH, model=model_arch)
  print("Generating heatmap...")
  try:
        iw = ip.execute_cams_pred(img_path)
        print("CAMs executed.")

        if iw.global_cams is None or len(iw.global_cams) == 0:
            print(f"No CAMs returned for {img_path}")
            return None

        cam = iw.global_cams[0]

        cam_resized = np.array(PIL.Image.fromarray(cam).resize((800, 800)))
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        cam_resized = np.expand_dims(cam_resized, axis=0)
        return cam_resized

  except Exception as e:
      print("Error during heatmap generation for", img_path, ":", str(e))
      raise e

def identify_dumps(image, size, k=1, dist=-1):
    """Identifies the coordinates of the most likely dump site from a heatmap."""
    if dist == -1:
        dist = max(size[0], size[1])
    pfx_sums = np.zeros(np.array(image.shape) + np.array([1, 1]))
    for i, x in enumerate(image):
        for j, y in enumerate(x):
            pfx_sums[i + 1, j + 1] = image[i, j] + pfx_sums[i + 1, j] + pfx_sums[i, j + 1] - pfx_sums[i, j]
    sorted_points = []
    for i in range(size[0] - 1, image.shape[0]):
        for j in range(size[1] - 1, image.shape[1]):
            count = pfx_sums[i + 1, j + 1] - pfx_sums[i + 1 - size[0], j + 1] - pfx_sums[i + 1, j + 1 - size[1]] + pfx_sums[i + 1 - size[0], j + 1 - size[1]]
            sorted_points.append((count, np.array([i - (size[0] - 1) // 2, j - (size[1] - 1) // 2])))
    sorted_points.sort(key=lambda x: -x[0])
    points = []
    for p in sorted_points:
        add = True
        for x in points:
            d = np.linalg.norm(p[1] - x)
            add = (d >= dist)
            if not add:
                break
        if add:
            points.append(p[1])
        if len(points) >= k:
            break
    return np.array(points)

def draw_bounding_boxes(height, width, x, y, image):
    """Crops an image based on center coordinates and box size."""
    x_min = int(x - width / 2)
    x_max = x_min + width
    y_min = int(y - height / 2)
    y_max = y_min + height
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

def call_model(img_path, bounding_box_size):
    """Main pipeline function that runs both models."""
    def preprocess_images(x_arr):
        transformations = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        processed = []
        for img in x_arr:
            img_pil = PIL.Image.fromarray(img.astype('uint8'))
            img_tensor = transformations(img_pil)
            processed.append(img_tensor)
        return torch.stack(processed)

    global classifier_model
    if classifier_model is None:
        print("Loading classification model for the first time...")
        # Using weights_only=False as it appears your model file requires it.
        torch.serialization.add_safe_globals([resnet.ResNet])
        classifier_model = torch.load("dump_classifier_full.pth", map_location="cpu", weights_only=False)
        classifier_model.to("cpu")

        classifier_model.eval()
        print("Classification model loaded.")

    # Step 1: Generate heatmap to find location
    heatmap = generate_heatmap(img_path)
    if heatmap is None:
        raise Exception("Heatmap generation failed.")

    # Step 2: Crop the original image based on the heatmap's brightest spot
    image = PIL.Image.open(img_path).convert("RGB")
    image_np = np.array(image)
    coords = identify_dumps(heatmap[0], bounding_box_size)
    if len(coords) == 0:
        raise Exception("Could not identify any dump sites in the heatmap.")
    coord = coords[0]  # (y, x)
    # Note: draw_bounding_boxes takes (height, width, x, y, image)
    box = draw_bounding_boxes(bounding_box_size[0], bounding_box_size[1], coord[1], coord[0], image_np)

    # Step 3: Classify the cropped image using the second model
    box_expanded = np.expand_dims(box, axis=0)
    box_processed = preprocess_images(box_expanded).to("cpu")


    with torch.no_grad():
        outputs = classifier_model(box_processed)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        probs_np = np.array(probs.cpu())

    print(f'Prediction Class: {pred_class}, Probabilities: {probs_np}')
    return (pred_class, probs_np, heatmap)

# --- Flask Web Server Code ---

app = Flask(__name__)
CORS(app)

def prepare_heatmap_for_response(heatmap_array):
    """Converts a numpy heatmap into a base64 encoded Data URL for the frontend."""
    heatmap_squeezed = np.squeeze(heatmap_array)
    colored_heatmap = cm.hot(heatmap_squeezed)
    img = PIL.Image.fromarray((colored_heatmap[:, :, :3] * 255).astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to handle image uploads and return model predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        try:
            # Run the full model pipeline
            bounding_box = np.array([100, 100])
            pred_class, probs, heatmap = call_model(filepath, bounding_box)
            heatmap_data_url = prepare_heatmap_for_response(heatmap)
            
            # Send the real results back to the frontend
            response_data = {
                'classIdx': int(pred_class),
                'confidences': probs.tolist(),
                'heatmap': heatmap_data_url
            }
            return jsonify(response_data)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Runs the server on port 5000
    app.run(debug=True, port=5000)