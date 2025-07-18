import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def enhance_contrast(path : str, destination_path : str, enhance_factor=1) -> np.ndarray:
    initial_img = Image.open(path)

    final_img = initial_img.copy()
    enhancer = ImageEnhance.Contrast(final_img)
    final_img = enhancer.enhance(enhance_factor)
    
    final_img.save(destination_path)
    return

def enhance_contrast(path : str, destination_path : str, enhance_factor=1) -> np.ndarray:
    initial_img = Image.open(path)

    final_img = initial_img.copy()
    enhancer = ImageEnhance.Sharpness(final_img)
    final_img = enhancer.enhance(enhance_factor)
    
    final_img.save(destination_path)
    return

def enhance_edge(path : str, destination_path : str) -> np.ndarray:
    initial_img = Image.open(path)

    final_image = initial_img.filter(ImageFilter.EDGE_ENHANCE)
    final_img.save(destination_path)
    return
