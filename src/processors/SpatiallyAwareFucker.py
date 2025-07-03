import cv2
import numpy as np
from scipy import ndimage
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from config.decorators import Decorators
from src.processors.Midas import Midas
from src.processors.SobelFilter import SobelFilter
from src.processors.UNetSaliency import UNetSaliency
from util.Processor import Processor


class SpatiallyAwareFusion(Processor):
    def __init__(self, image, predictor=None, importance_weights=None):
        """
        Spatially-aware fusion that preserves object spatial relationships

        Args:
            image: Input image
            predictor: Detectron2 predictor for object detection (optional)
            importance_weights: Weights for different object importance levels
        """
        super().__init__(image)
        self.predictor = predictor
        self.importance_weights = importance_weights or {
            'high': 0.8,  # Critical objects (people, faces, main subjects)
            'medium': 0.6,  # Important objects (vehicles, furniture)
            'low': 0.4  # Background objects
        }
        self.epsilon = 1e-8

    def setup_object_detector(self, config_path=None, weights_path=None):
        """Setup Detectron2 object detector if not provided"""
        if self.predictor is None and config_path and weights_path:
            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            cfg.MODEL.WEIGHTS = weights_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Lower threshold for more objects
            self.predictor = DefaultPredictor(cfg)

    @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        # Generate base feature maps
        depth = Midas(self._image)().image()
        saliency = UNetSaliency(self._image)().image()
        gradient = SobelFilter(self._image)().image()

        # Apply spatially-aware fusion
        self._image = self.spatial_fusion_with_object_preservation(depth, saliency, gradient)

    def detect_and_classify_objects(self, image):
        """
        Detect objects and classify them by importance for retargeting

        Returns:
            dict: Object information with spatial masks and importance levels
        """
        if self.predictor is None:
            # Fallback: use saliency-based object detection
            return self.fallback_object_detection(image)

        outputs = self.predictor(image)

        if len(outputs["instances"]) == 0:
            return {"objects": [], "combined_mask": np.zeros(image.shape[:2])}

        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        masks = outputs["instances"].pred_masks.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()

        objects = []
        combined_importance_mask = np.zeros(image.shape[:2])

        for i, (box, mask, cls, score) in enumerate(zip(boxes, masks, classes, scores)):
            # Classify object importance based on class and other factors
            importance = self.classify_object_importance(cls, box, mask, score, image.shape)

            obj_info = {
                'id': i,
                'bbox': box,
                'mask': mask,
                'class': cls,
                'score': score,
                'importance': importance,
                'center': self.calculate_object_center(box),
                'size_ratio': self.calculate_size_ratio(box, image.shape)
            }
            objects.append(obj_info)

            # Add to combined importance mask
            combined_importance_mask += mask * self.importance_weights[importance]

        # Normalize combined mask
        if np.max(combined_importance_mask) > 0:
            combined_importance_mask = combined_importance_mask / np.max(combined_importance_mask)

        return {
            "objects": objects,
            "combined_mask": combined_importance_mask
        }

    def classify_object_importance(self, class_id, bbox, mask, score, image_shape):
        """
        Classify object importance based on multiple factors
        """
        # Common object class mappings (COCO dataset)
        high_importance_classes = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
        medium_importance_classes = [6, 8, 9, 10, 11, 13, 14, 15, 16]  # train, boat, etc.

        # Factor 1: Object class
        if class_id in high_importance_classes:
            class_importance = 'high'
        elif class_id in medium_importance_classes:
            class_importance = 'medium'
        else:
            class_importance = 'low'

        # Factor 2: Object size (larger objects are more important)
        size_ratio = self.calculate_size_ratio(bbox, image_shape)

        # Factor 3: Central position (objects near center are more important)
        center = self.calculate_object_center(bbox)
        image_center = np.array([image_shape[1] / 2, image_shape[0] / 2])
        center_distance = np.linalg.norm(center - image_center) / max(image_shape[:2])

        # Factor 4: Detection confidence
        confidence_factor = min(score, 1.0)

        # Combine factors to adjust importance
        adjustment_score = (size_ratio * 0.4 + (1 - center_distance) * 0.3 + confidence_factor * 0.3)

        # Adjust importance based on combined score
        if adjustment_score > 0.7 and class_importance != 'low':
            return 'high'
        elif adjustment_score > 0.4:
            return class_importance
        else:
            # Downgrade importance for small, peripheral, low-confidence objects
            importance_levels = ['high', 'medium', 'low']
            current_idx = importance_levels.index(class_importance)
            return importance_levels[min(current_idx + 1, 2)]

    def calculate_object_center(self, bbox):
        """Calculate normalized center of bounding box"""
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    def calculate_size_ratio(self, bbox, image_shape):
        """Calculate object size relative to image"""
        obj_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        image_area = image_shape[0] * image_shape[1]
        return obj_area / image_area

    def fallback_object_detection(self, image):
        """Fallback object detection using saliency when Detectron2 is not available"""
        saliency = UNetSaliency(self._image)().image()

        # Threshold saliency to find object regions
        threshold = np.percentile(saliency, 80)
        binary_mask = saliency > threshold

        # Find connected components
        labeled, num_features = ndimage.label(binary_mask)

        objects = []
        combined_mask = np.zeros_like(saliency)

        for i in range(1, num_features + 1):
            mask = labeled == i

            # Filter small objects
            if np.sum(mask) < 0.001 * mask.size:
                continue

            # Create bounding box
            coords = np.where(mask)
            bbox = [np.min(coords[1]), np.min(coords[0]),
                    np.max(coords[1]), np.max(coords[0])]

            # Estimate importance based on saliency strength and size
            avg_saliency = np.mean(saliency[mask])
            size_ratio = np.sum(mask) / mask.size

            if avg_saliency > 0.7 and size_ratio > 0.01:
                importance = 'high'
            elif avg_saliency > 0.5 or size_ratio > 0.005:
                importance = 'medium'
            else:
                importance = 'low'

            obj_info = {
                'id': i,
                'bbox': bbox,
                'mask': mask,
                'importance': importance,
                'center': self.calculate_object_center(bbox),
                'size_ratio': size_ratio
            }
            objects.append(obj_info)

            combined_mask += mask * self.importance_weights[importance]

        # Normalize
        if np.max(combined_mask) > 0:
            combined_mask = combined_mask / np.max(combined_mask)

        return {"objects": objects, "combined_mask": combined_mask}

    def create_spatial_preservation_maps(self, objects, image_shape):
        """
        Create spatial maps that preserve object relationships

        Returns:
            dict: Different preservation maps for different fusion aspects
        """
        h, w = image_shape[:2]

        # 1. Object importance map (high values = important objects)
        importance_map = np.zeros((h, w))

        # 2. Spatial relationship map (preserves relative positions)
        spatial_relationship_map = np.zeros((h, w))

        # 3. Object boundary preservation map (high values = object edges)
        boundary_map = np.zeros((h, w))

        # 4. Size preservation map (prevents critical objects from shrinking too much)
        size_preservation_map = np.zeros((h, w))

        for obj in objects:
            mask = obj['mask']
            importance_weight = self.importance_weights[obj['importance']]

            # Fill importance map
            importance_map += mask * importance_weight

            # Create spatial relationship preservation
            # Objects near image center get higher preservation priority
            center_dist_map = self.create_center_distance_map(obj['center'], (h, w))
            spatial_relationship_map += mask * importance_weight * (1 - center_dist_map)

            # Create boundary preservation map
            boundary = self.extract_object_boundary(mask)
            boundary_map += boundary * importance_weight * 1.5  # Boost boundaries

            # Size preservation (larger objects get more protection)
            size_factor = min(obj['size_ratio'] * 10, 1.0)  # Scale up small ratios
            size_preservation_map += mask * importance_weight * size_factor

        # Normalize all maps
        maps = {
            'importance': self.normalize_map(importance_map),
            'spatial_relationship': self.normalize_map(spatial_relationship_map),
            'boundary': self.normalize_map(boundary_map),
            'size_preservation': self.normalize_map(size_preservation_map)
        }

        return maps

    def create_center_distance_map(self, object_center, image_shape):
        """Create a map of distances from object center"""
        h, w = image_shape
        y, x = np.ogrid[:h, :w]
        center_x, center_y = object_center

        # Normalized distance from object center
        dist_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(w ** 2 + h ** 2)

        return dist_map / max_dist

    def extract_object_boundary(self, mask, dilation_size=3):
        """Extract object boundaries using morphological operations"""
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = dilated - eroded
        return boundary.astype(bool)

    def spatial_fusion_with_object_preservation(self, depth, saliency, gradient):
        """
        Main fusion function that preserves object spatial relationships
        """
        # Normalize input maps
        depth_norm = self.normalize_map(1.0 - depth)  # Invert depth for proximity
        saliency_norm = self.normalize_map(saliency)
        gradient_norm = self.normalize_map(gradient)

        # Detect objects and get preservation maps
        object_info = self.detect_and_classify_objects(self._image.rgb())
        preservation_maps = self.create_spatial_preservation_maps(
            object_info["objects"], self._image.rgb().shape
        )

        print(f"Detected {len(object_info['objects'])} objects for spatial preservation")

        # Create spatially-adaptive weights based on object preservation needs
        adaptive_weights = self.compute_spatial_adaptive_weights(
            depth_norm, saliency_norm, gradient_norm, preservation_maps
        )

        # Apply object-aware fusion
        fused_result = (adaptive_weights['depth'] * depth_norm +
                        adaptive_weights['saliency'] * saliency_norm +
                        adaptive_weights['gradient'] * gradient_norm)

        # Post-process to enhance object preservation
        enhanced_result = self.enhance_object_preservation(
            fused_result, preservation_maps, object_info
        )

        return self.normalize_map(enhanced_result)

    def compute_spatial_adaptive_weights(self, depth, saliency, gradient, preservation_maps):
        """
        Compute spatially-varying weights that prioritize object preservation
        """
        h, w = depth.shape

        # Base weights (your original strategy)
        base_depth_weight = 0.5
        base_saliency_weight = 0.35
        base_gradient_weight = 0.15

        # Initialize adaptive weight maps
        depth_weights = np.full((h, w), base_depth_weight)
        saliency_weights = np.full((h, w), base_saliency_weight)
        gradient_weights = np.full((h, w), base_gradient_weight)

        # Apply spatial modulations based on preservation needs
        importance_map = preservation_maps['importance']
        boundary_map = preservation_maps['boundary']
        spatial_map = preservation_maps['spatial_relationship']

        # Modulation 1: Boost depth weights for important near objects
        near_important_objects = (depth > 0.6) * (importance_map > 0.5)
        depth_weights += near_important_objects * 0.2  # Up to 20% boost

        # Modulation 2: Boost saliency weights for spatially important regions
        saliency_weights += spatial_map * 0.3  # Up to 30% boost

        # Modulation 3: Boost gradient weights at object boundaries
        gradient_weights += boundary_map * 0.4  # Up to 40% boost for boundaries

        # Modulation 4: Special handling for high-importance objects
        critical_regions = importance_map > 0.7
        depth_weights[critical_regions] *= 1.3
        saliency_weights[critical_regions] *= 1.2

        # Normalize weights spatially
        total_weights = depth_weights + saliency_weights + gradient_weights
        depth_weights /= (total_weights + self.epsilon)
        saliency_weights /= (total_weights + self.epsilon)
        gradient_weights /= (total_weights + self.epsilon)

        return {
            'depth': depth_weights,
            'saliency': saliency_weights,
            'gradient': gradient_weights
        }

    def enhance_object_preservation(self, fused_result, preservation_maps, object_info):
        """
        Post-process the fused result to enhance object preservation
        """
        enhanced = fused_result.copy()

        # Enhancement 1: Boost object cores
        object_cores = preservation_maps['importance'] > 0.6
        enhanced[object_cores] = np.minimum(enhanced[object_cores] * 1.15, 1.0)

        # Enhancement 2: Strengthen object boundaries
        boundaries = preservation_maps['boundary'] > 0.3
        enhanced[boundaries] = np.minimum(enhanced[boundaries] * 1.1, 1.0)

        # Enhancement 3: Create protective "halos" around critical objects
        for obj in object_info["objects"]:
            if obj['importance'] == 'high':
                # Create a protective region around high-importance objects
                dilated_mask = cv2.dilate(obj['mask'].astype(np.uint8),
                                          np.ones((7, 7), np.uint8), iterations=1)
                halo = dilated_mask - obj['mask'].astype(np.uint8)
                enhanced += halo * 0.05  # Small boost to surrounding area

        return np.clip(enhanced, 0, 1)

    def normalize_map(self, map_data):
        """Robust normalization"""
        if map_data is None or map_data.size == 0:
            return np.zeros_like(map_data) if map_data is not None else None

        p1, p99 = np.percentile(map_data, [1, 99])
        clipped = np.clip(map_data, p1, p99)

        min_val = np.min(clipped)
        max_val = np.max(clipped)

        if max_val - min_val > self.epsilon:
            return (clipped - min_val) / (max_val - min_val)
        else:
            return np.full_like(clipped, 0.5)


# Quality assessment integration
def compute_retargeting_quality_with_spatial_preservation(original_img, retargeted_img, predictor=None):
    """
    Compute quality score that considers spatial preservation from the document
    """

    # Use the approach from your document
    def normalize_position(box, image_shape):
        height, width = image_shape[:2]
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        return np.array([x_center / width, y_center / height])

    def compute_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    if predictor is None:
        print("Warning: No object detector provided, using fallback quality assessment")
        return 0.5  # Fallback score

    # Detect objects in both images
    orig_outputs = predictor(original_img)
    retar_outputs = predictor(retargeted_img)

    if len(orig_outputs["instances"]) == 0 or len(retar_outputs["instances"]) == 0:
        return 0.0

    orig_boxes = orig_outputs["instances"].pred_boxes.tensor.cpu().numpy()
    orig_masks = orig_outputs["instances"].pred_masks.cpu().numpy()
    retar_boxes = retar_outputs["instances"].pred_boxes.tensor.cpu().numpy()
    retar_masks = retar_outputs["instances"].pred_masks.cpu().numpy()

    # Match objects between original and retargeted images
    quality_scores = []
    for i in range(min(len(orig_boxes), len(retar_boxes))):
        # Position preservation
        orig_pos = normalize_position(orig_boxes[i], original_img.shape)
        retar_pos = normalize_position(retar_boxes[i], retargeted_img.shape)
        position_shift = np.linalg.norm(orig_pos - retar_pos)

        # Object preservation (IoU)
        iou = compute_iou(orig_masks[i], retar_masks[i])

        # Combined score (from document formula)
        w1, w2 = 0.4, 0.6  # Weights favoring object preservation over position
        quality_score = w1 * (1 - position_shift) + w2 * iou
        quality_scores.append(quality_score)

    return np.mean(quality_scores) if quality_scores else 0.0