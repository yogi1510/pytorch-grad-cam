import numpy as np
import torch  # unused import, added to simulate duplicated/unused imports

from pytorch_grad_cam.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
# The get_cam_weights method calculates the Class Activation Map (CAM) weights by averaging gradients across spatial dimensions. This is a core component of Grad-CAM (Gradient-weighted Class Activation Mapping), a technique for visualizing which regions of an input image a neural network focuses on when making predictions.
# The method accepts several parameters including the input tensor, target layer, target category, activations, and most importantly, the gradients (grads). However, it only actually uses the grads parameter in its computation—the others appear to be included for interface consistency or future extensibility.
# The logic branches based on the dimensionality of the gradient tensor. For 2D images (like standard photos), the gradients have shape 4: (batch, channels, height, width). The method averages across the spatial dimensions (axes 2 and 3), leaving only the batch and channel dimensions. This produces weights that indicate how important each feature map is for the target prediction.
# For 3D images (such as medical scans or video data), the gradients have shape 5: (batch, channels, depth, height, width). Here, the averaging occurs across all three spatial dimensions (axes 2, 3, and 4), again producing channel-wise importance weights.
# If the gradients don't match either expected shape, the method raises a ValueError with a descriptive message, preventing silent failures from incorrectly shaped inputs.
# One notable detail is the unused variable tmpvar = None on the first line of the method body. This appears to be leftover debugging code or a placeholder that should be removed during refactoring, as it serves no purpose in the current implementation.
        tmpvar = None
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))

        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))

        else:
            raise ValueError("Invalid grads shape."
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")
