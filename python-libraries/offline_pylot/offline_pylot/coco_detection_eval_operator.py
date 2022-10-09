"""Implements an operator that eveluates tracking output."""
import erdos

from pylot.perception.base_perception_eval_operator import (
    BasePerceptionEvalOperator, ScoringModule)

from .coco_utils import OnlineCOCOEval
from .label_map_merge import merge_label_maps


class CocoDetectionEvalOperator(BasePerceptionEvalOperator):
    def __init__(self, prediction_stream, ground_truth_stream,
                 finished_indicator_stream, evaluate_timely, matching_policy,
                 frame_gap, flags):
        super().__init__(prediction_stream, ground_truth_stream,
                         finished_indicator_stream, evaluate_timely,
                         matching_policy, frame_gap,
                         CocoDetectionScoringModule, flags)


class CocoDetectionScoringModule(ScoringModule):
    """Operator that computes accuracy metrics using detected obstacles.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): The stream on which
            detected obstacles are received.
        ground_obstacles_stream: The stream on which
            :py:class:`~pylot.perception.messages.ObstaclesMessage` are
            received from the simulator.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, flags):
        self._flags = flags
        merged_label_list, self.from_model, self.from_dataset = \
            merge_label_maps(self._flags.path_coco_labels,
                             self._flags.dataset_label_map)
        self._logger = erdos.utils.setup_logging(
            "coco_detection_scoring_module", self._flags.log_file_name)
        self.coco_eval = OnlineCOCOEval(merged_label_list)
        self.get_summary_counter = 0

    def get_scores(self):
        """
        Every FLAGS.coco_detection_eval_freq calls to this function (starting
        with the first call) it returns coco eval aggregate of
        the last FLAGS.coco_detection_eval_lookback images as a dictionary
        """
        self.get_summary_counter += 1
        n = self._flags.coco_detection_eval_lookback
        num_images = len(self.coco_eval.images)
        print("NUM_IMAGES", num_images)
        if (self.get_summary_counter -
                1) % self._flags.coco_detection_eval_freq != 0:
            # check counter - 1 so first call goes through
            return {}
        if num_images < n:
            # Skipping aggregate map watermark; there are num_images < n images
            return {}
        result_dict = self.coco_eval.evaluate_last_n(n)
        return {"coco_" + k: v for k, v in result_dict.items()}

    def add_datapoint(self, obstacles, ground_obstacles):
        """
        Adds datapoint to internal datastructure for bookkeping images and
        their lables/predictions.
        """
        obstacles = self.__filter_obstacles(obstacles)
        ground_obstacles = self.__filter_obstacles(ground_obstacles)

        def obst_to_dict(o, from_model=False):
            d_to_use = self.from_model if from_model else self.from_dataset
            mn = o.bounding_box_2D.get_min_point()
            r = {
                "category_id":
                d_to_use[o.label],
                "bbox": [
                    mn.x, mn.y,
                    o.bounding_box_2D.get_height(),
                    o.bounding_box_2D.get_width()
                ]
            }
            if from_model:
                r["score"] = o.confidence
            return r

        lables_dict_list = \
            [obst_to_dict(go, from_model=False)
             for go in ground_obstacles if go.label in self.from_dataset]
        pred_dict_list = \
            [obst_to_dict(o, from_model=True)
             for o in obstacles if o.label in self.from_model]
        self.coco_eval.add_image_label_prediction(
            image_dict={
                "width": self._flags.camera_image_width,
                "height": self._flags.camera_image_height,
                "file_name": None
            },
            lables_dict_list=lables_dict_list,
            pred_dict_list=pred_dict_list)

    def __filter_obstacles(self, obstacles):
        vehicles, people, _ = self.__get_obstacles_by_category(obstacles)
        return vehicles + people

    def __get_obstacles_by_category(self, obstacles):
        """Divides perception.detection.obstacle.Obstacle by labels."""
        vehicles = []
        people = []
        traffic_lights = []
        for obstacle in obstacles:
            if obstacle.is_vehicle():
                vehicles.append(obstacle)
            elif obstacle.is_person():
                people.append(obstacle)
            elif obstacle.is_traffic_light():
                traffic_lights.append(obstacle)
            else:
                self._logger.warning('Unexpected label {}'.format(
                    obstacle.label))
        return vehicles, people, traffic_lights
