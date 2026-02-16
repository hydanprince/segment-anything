import numpy as np

def crop_muzzle(image, predictor, input_points, input_labels):
    """
    Crops the muzzle region using SAM predictor.
    
    Args:
        image: RGB image (numpy array)
        predictor: SamPredictor object
        input_points: np.array of prompt points [[x1, y1], [x2, y2], ...]
        input_labels: np.array of labels (1=positive, 0=negative)
        
    Returns:
        cropped muzzle image (numpy array)
    """
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    mask = masks[np.argmax(scores)]
    
    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    
    cropped = image[y1:y2, x1:x2]
    return cropped
