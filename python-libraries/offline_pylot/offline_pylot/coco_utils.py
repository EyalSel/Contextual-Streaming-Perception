import json
from datetime import datetime

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .utils import verify_keys_in_dict

# groundtruth_dataset_template = {
#     "info": {
#         "version": "0.0.1",
#         "description": "AD_evaluation",
#         "date_created": datetime.now(),
#     },
#     "images": [
#         {
#             "id": int, "width": int, "height": int, "file_name": str,
#         }
#     ],
#     "annotations": [
#         {
#             "id": int, "image_id": int, "category_id": int, "area": float,
#             "bbox": [x, y, width, height], "iscrowd": 0
#         }
#     ],
#     "categories": [
#         {
#             "id": int, "name": str, "supercategory": str
#         }
#     ],
#     "licenses": None
# }

# predictions_template = [
#     {
#         "image_id": 42, "category_id": 18,
#         "bbox": [258.15, 41.29, 348.26, 243.78],
#         "score":0.236
#     }
# ]


class OnlineCOCOEval:
    def __init__(self, label_list):
        assert type(label_list) == list, label_list
        self.categories = [{
            "id": i,
            "name": l.strip(),
            "supercategory": "AD_object"
        } for i, l in enumerate(label_list)]
        self.model_label_map = {x["name"]: x["id"] for x in self.categories}
        self.next_image_id = 0
        self.next_annotation_id = 1
        self.images = []
        self.annotations = []
        self.preds = []

    def add_image_label_prediction(self, image_dict, lables_dict_list,
                                   pred_dict_list):
        """
        image_dict: {"width": , "height: ,"file_name":}
        lables_dict_list: [{"category_id": , "bbox": [x,y,w,h] }, ...]
        pred_dict_list: [{"category_id": , "bbox": [x,y,w,h], "score": }, ...]
        """
        img_id = self.next_image_id
        self.next_image_id += 1
        verify_keys_in_dict(["width", "height", "file_name"], image_dict)
        image_dict["id"] = img_id

        def prep_label_dict(label_dict):
            verify_keys_in_dict(["category_id", "bbox"], label_dict)
            error_msg = ("The dataset uses a label {} that doesn't show up "
                         "in the model's label map {}")
            assert label_dict["category_id"] in self.model_label_map, \
                error_msg.format(label_dict["category_id"],
                                 self.model_label_map)
            label_dict["category_id"] = self.model_label_map[
                label_dict["category_id"]]
            label_dict["image_id"] = img_id
            label_dict["area"] = label_dict["bbox"][2] * label_dict["bbox"][3]
            label_dict["iscrowd"] = 0
            label_dict["id"] = self.next_annotation_id
            self.next_annotation_id += 1
            return label_dict

        lables_dict_list = [prep_label_dict(d) for d in lables_dict_list]
        lables_dict_list = filter(lambda x: x is not None, lables_dict_list)

        def prep_pred_dict(pred_dict):
            verify_keys_in_dict(["category_id", "bbox", "score"], pred_dict)
            error_msg = ("given model prediction {} that is not in the "
                         "predefined model label map {}")
            assert pred_dict["category_id"] in self.model_label_map, \
                error_msg.format(pred_dict["category_id"],
                                 self.model_label_map)
            pred_dict["category_id"] = self.model_label_map[
                pred_dict["category_id"]]
            pred_dict["image_id"] = img_id
            return pred_dict

        pred_dict_list = [prep_pred_dict(d) for d in pred_dict_list]

        self.images.append(image_dict)
        self.annotations.extend(lables_dict_list)
        self.preds.extend(pred_dict_list)

    def evaluate_last_n(self, n=None, verbose=False):
        """
        if n = `None` evaluate over all images added so far
        """
        assert n is None or n > 0, "Should evaluate over at least 1 image"
        assert len(self.images) > 0, "No images to evaluate on"
        # assert len(self.annotations) > 0, "No annotations to evaluate on"
        # assert len(self.preds) > 0, "No predictions to evaluate on"
        n = -len(self.images) if n is None else -n
        images_to_use = self.images[n:]
        preds_to_use = [
            p for p in self.preds if p["image_id"] >= images_to_use[0]["id"]
        ]
        anns_to_use = [
            a for a in self.annotations
            if a["image_id"] >= images_to_use[0]["id"]
        ]
        groundtruth_dataset_template = {
            "info": {
                "version": "0.0.1",
                "description": "AD_evaluation",
                "date_created": datetime.now().isoformat(),
            },
            "images": images_to_use,
            "annotations": anns_to_use,
            "categories": self.categories
        }
        # labels_categories = set([a["category_id"] for a in anns_to_use])
        # pred_categories = set([a["category_id"] for a in preds_to_use])
        # preds_not_in_labels = pred_categories.difference(labels_categories)
        # assert len(preds_not_in_labels) == 0, \
        #     "The model predicts categories {} that don't show up in the dataset".format(([self.categories[idx] for idx in labels_categories],   # noqa: E501
        #     [self.categories[idx] for idx in pred_categories]))
        keys = [
            "AP_IoU=0.50:0.95_area=all_maxDets=100",
            "AP_IoU=0.50_area=all_maxDets=100",
            "AP_IoU=0.75_area=all_maxDets=100",
            "AP_IoU=0.50:0.95_area=small_maxDets=100",
            "AP_IoU=0.50:0.95_area=medium_maxDets=100",
            "AP_IoU=0.50:0.95_area=large_maxDets=100",
            "AR_IoU=0.50:0.95_area=all_maxDets=1",
            "AR_IoU=0.50:0.95_area=all_maxDets=10",
            "AR_IoU=0.50:0.95_area=all_maxDets=100",
            "AR_IoU=0.50:0.95_area=small_maxDets=100",
            "AR_IoU=0.50:0.95_area=medium_maxDets=100",
            "AR_IoU=0.50:0.95_area=large_maxDets=100",
        ]
        if verbose:
            print(
                json.dumps(groundtruth_dataset_template,
                           indent=4,
                           sort_keys=True))
            print(json.dumps(preds_to_use, indent=4, sort_keys=True))
        if len(preds_to_use) == 0 and len(anns_to_use) > 0:
            return {k: 0 for k in keys}
        elif len(preds_to_use) == 0 and len(anns_to_use) == 0:
            return {k: -1 for k in keys}
        cocoGt = COCO()
        cocoGt.dataset = groundtruth_dataset_template
        cocoGt.createIndex()
        cocoDt = cocoGt.loadRes(preds_to_use)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        # cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        values = cocoEval.stats
        return {k: v for k, v in zip(keys, values)}
