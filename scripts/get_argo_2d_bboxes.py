# Based on the script from: https://github.com/argoai/argoverse-api/issues/144

import argparse
import copy
import glob
import logging
import os
import pickle
from pathlib import Path
from typing import Any, List, Sequence, Union

import PIL.Image as Image

from argoverse.data_loading.object_label_record import \
    json_label_dict_to_obj_record
from argoverse.data_loading.simple_track_dataloader import \
    SimpleArgoverseTrackingDataLoader
from argoverse.utils.calibration import (
    Calibration, get_calibration_config, point_cloud_to_homogeneous,
    project_lidar_to_img_motion_compensated)
from argoverse.utils.camera_stats import (RING_CAMERA_LIST, RING_IMG_HEIGHT,
                                          RING_IMG_WIDTH, STEREO_CAMERA_LIST)
from argoverse.utils.ffmpeg_utils import write_nonsequential_idx_video
from argoverse.utils.frustum_clipping import (cuboid_to_2d_frustum_bbox,
                                              generate_frustum_planes)
from argoverse.utils.ply_loader import load_ply

import cv2

import imageio

import matplotlib.pyplot as plt

import numpy as np

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

#: Any numeric type
Number = Union[int, float]

name_of_camera = 'ring_front_center'


def overlaps(bbox1_xmin,
             bbox1_ymin,
             bbox1_xmax,
             bbox1_ymax,
             bbox2_xmin,
             bbox2_ymin,
             bbox2_xmax,
             bbox2_ymax,
             threshold=0.66):
    #     print (bbox1_xmin, bbox1_ymin, bbox1_xmax, bbox1_ymax, bbox2_xmin, bbox2_ymin, bbox2_xmax, bbox2_ymax)
    if bbox1_xmin > bbox2_xmax or bbox2_xmin > bbox1_xmax:  # gap in x
        return False
    elif bbox2_ymin > bbox2_ymax or bbox2_ymin > bbox1_ymax:  # gap in y
        return False
    else:
        area1 = (bbox1_xmax - bbox1_xmin) * (bbox1_ymax - bbox1_ymin)
        #         area2 = (bbox2_xmax - bbox2_xmin)*(bbox2_ymax - bbox2_ymin)
        xs = [bbox1_xmin, bbox1_xmax, bbox2_xmin, bbox2_xmax]
        ys = [bbox1_ymin, bbox1_ymax, bbox2_ymin, bbox2_ymax]
        xs.sort()
        ys.sort()
        area_overlap = (xs[2] - xs[1]) * (
            ys[2] - ys[1])  # interior points are always the overlaping ones
        #         if max(area_overlap / area1 , area_overlap /area2) > threshold:
        if (area_overlap / area1) > threshold:
            return True
        else:
            return False


def shrink_box(bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax):
    x_avg = (bbox_xmin + bbox_xmax) / 2
    y_avg = (bbox_ymin + bbox_ymax) / 2
    return ((bbox_xmin + x_avg) / 2, (bbox_ymin + y_avg) / 2,
            (bbox_xmax + x_avg) / 2, (bbox_ymax + y_avg) / 2)


def ordering(uv1, xmin, ymin, xmax, ymax, cuboid=None):
    if isinstance(cuboid, np.ndarray):
        return np.linalg.norm(cuboid)

    uv1 = uv1.T
    xmin, ymin, xmax, ymax = shrink_box(xmin, ymin, xmax, ymax)
    lidar_bbox = uv1[(xmin < uv1[:, 0]) & (uv1[:, 0] < xmax) &
                     (ymin < uv1[:, 1]) & (uv1[:, 1] < ymax)]
    if len(lidar_bbox[:, 2]) != 0:
        return np.mean(lidar_bbox[:, 2])
    else:
        return 500


def pickle_format(bboxes, img):
    bboxes = [{
        'bbox': [xmin, xmax, ymin, ymax],
        'label': label,
        'id': track_id
    } for (xmin, ymin), (xmax, ymax), track_id, label, in bboxes]
    return {'center_camera_feed': np.array(img), 'obstacles': bboxes}


def plot_img_2d_bboxes(labels,
                       planes,
                       img: np.ndarray,
                       log_calib_data,
                       camera_name: str,
                       cam_timestamp: int,
                       lidar_timestamp: int,
                       data_dir: str,
                       log_id: str,
                       save_img_fpath: str,
                       lidar_pts,
                       id_dict,
                       video=False,
                       show_depthmap=False,
                       cuboid=True,
                       cache_all_bbox=True,
                       cache_img=False):
    """ """
    cam_config = get_calibration_config(log_calib_data, camera_name)
    calib_cam = next(
        (c for c in log_calib_data["camera_data_"]
         if c["key"] == f"image_raw_{camera_name}"),
        None,
    )
    if calib_cam is None:
        print(f"No Camera of name: {camera_name}")
    calib = Calibration(cam_config, calib_cam)
    pc = lidar_pts
    uv = calib.project_ego_to_image(pc).T

    idx_ = np.where(
        np.logical_and.reduce(
            (uv[0, :] >= 0.0, uv[0, :] < np.shape(img)[1] - 1.0,
             uv[1, :] >= 0.0, uv[1, :] < np.shape(img)[0] - 1.0,
             uv[2, :] > 0)))
    idx_ = idx_[0]

    uv1 = uv[:, idx_]
    if uv1 is None:
        raise Exception('No point image projection')

    bboxes = []
    cuboid_centers = []
    for label_idx, label in enumerate(labels):
        # print (type(label))
        obj_rec = json_label_dict_to_obj_record(label)
        track_id = (obj_rec.track_id).replace("-", "")
        #         track_id_short = int(track_id,16) % sys.maxsize
        track_id_short = id_dict[track_id]
        if obj_rec.label_class in [
                "ANIMAL", "STROLLER", "OTHER_MOVER", "ON_ROAD_OBSTACLE"
        ]:
            continue
        if obj_rec.occlusion == 100:
            continue
        if obj_rec.occlusion > 0:
            print("Occlusion {}".format(obj_rec.occlusion))

        cuboid_vertices = obj_rec.as_3d_bbox()
        points_h = point_cloud_to_homogeneous(cuboid_vertices).T

        uv, uv_cam, valid_pts_bool, camera_config = project_lidar_to_img_motion_compensated(
            points_h,  # these are recorded at lidar_time
            copy.deepcopy(log_calib_data),
            camera_name,
            cam_timestamp,
            lidar_timestamp,
            data_dir,
            log_id,
            return_K=True,
        )
        K = camera_config.intrinsic

        if valid_pts_bool.sum() == 0:
            continue
        bbox_2d = cuboid_to_2d_frustum_bbox(uv_cam.T[:, :3], planes, K[:3, :3])
        cuboid_center = (uv_cam.T[:, :3]).mean(axis=0)
        if bbox_2d is None:
            continue
        else:
            x1, y1, x2, y2 = bbox_2d

            x1 = min(x1, RING_IMG_WIDTH - 1)
            x2 = min(x2, RING_IMG_WIDTH - 1)
            y1 = min(y1, RING_IMG_HEIGHT - 1)
            y2 = min(y2, RING_IMG_HEIGHT - 1)

            x1 = max(x1, 0)
            x2 = max(x2, 0)
            y1 = max(y1, 0)
            y2 = max(y2, 0)

            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            start = (int(xmin), int(ymin))
            end = (int(xmax), int(ymax))
            if xmin != xmax and ymin != ymax:  # filter out zero volume boxes
                bboxes.append(
                    (start, end, track_id_short, obj_rec.label_class))
                cuboid_centers.append(cuboid_center)

    average_depth_lidar = [
        ordering(uv1, xmin, ymin, xmax, ymax)
        for (xmin, ymin), (xmax, ymax), _, _ in bboxes
    ]
    average_depth_cuboid = [
        ordering(uv1, xmin, ymin, xmax, ymax, cuboid)
        for ((xmin, ymin), (xmax, ymax), _,
             _), cuboid in zip(bboxes, cuboid_centers)
    ]
    if not cuboid:
        average_depth = average_depth_lidar
    else:
        average_depth = average_depth_cuboid

    new_bboxes = []
    for i in range(len(bboxes)):
        (bbox1_xmin,
         bbox1_ymin), (bbox1_xmax,
                       bbox1_ymax), bbox1_id, bbox1_label = bboxes[i]
        is_occluded = False
        for j in range(len(bboxes)):
            if i != j:
                (bbox2_xmin, bbox2_ymin), (bbox2_xmax,
                                           bbox2_ymax), _, _ = bboxes[j]
                if overlaps(bbox1_xmin, bbox1_ymin, bbox1_xmax, bbox1_ymax,
                            bbox2_xmin, bbox2_ymin, bbox2_xmax, bbox2_ymax):
                    is_occluded = is_occluded or (average_depth[j] <
                                                  average_depth[i])

        if not is_occluded:
            new_bboxes.append(
                ((bbox1_xmin, bbox1_ymin), (bbox1_xmax, bbox1_ymax), bbox1_id,
                 bbox1_label))

    if video or show_depthmap:
        img = Image.fromarray(img[:, :, ::-1])
    if video:
        img.save(save_img_fpath)

    if show_depthmap:
        plt.imshow(img)
        cm = plt.cm.get_cmap('jet')
        plt.scatter(uv1[0], uv1[1], c=1 - uv1[2] / max(uv1[2]), s=1, cmap=cm)
        plt.axis('off')
        plt.savefig("base_" + save_img_fpath)
        plt.clf()

    out = pickle_format(new_bboxes, img)

    if cache_all_bbox:
        out["all_bbox"] = [(bbox, lidar, cuboid) for (
            bbox, lidar,
            cuboid) in zip(bboxes, average_depth_lidar, average_depth_cuboid)]

    return out


def dump_log_2d_bboxes_to_imgs(
    log_ids: Sequence[str],
    max_num_images_to_render: int,
    data_dir: str,
    experiment_prefix: str,
    video: bool,
    motion_compensate: bool = True,
) -> List[str]:
    """
    We bring the 3D points into each camera coordinate system, and do the clipping there in 3D.

    Args:
        log_ids: A list of log IDs
        max_num_images_to_render: maximum numbers of images to render.
        data_dir: path to dataset with the latest data
        experiment_prefix: Output directory
        motion_compensate: Whether to motion compensate when projecting

    Returns:
        saved_img_fpaths
    """
    dl = SimpleArgoverseTrackingDataLoader(data_dir=data_dir,
                                           labels_dir=data_dir)

    for log_id in log_ids:
        save_dir = f"{experiment_prefix}_{log_id}"
        if video:
            if not Path(save_dir).exists():
                os.makedirs(save_dir)

        log_calib_data = dl.get_log_calibration_data(log_id)

        for cam_idx, camera_name in enumerate(RING_CAMERA_LIST +
                                              STEREO_CAMERA_LIST):

            if camera_name != name_of_camera:
                continue

            # print (dl.data_dir, log_id,camera_name)
            cam_im_fpaths = dl.get_ordered_log_cam_fpaths(log_id, camera_name)
            # print (len(cam_im_fpaths), log_id, camera_name)

            # BUILD LOOKUP TABLE OF IDS
            ids = []
            for i, im_fpath in enumerate(cam_im_fpaths):
                cam_timestamp = Path(im_fpath).stem.split("_")[-1]
                cam_timestamp = int(cam_timestamp)
                ply_fpath = dl.get_closest_lidar_fpath(log_id, cam_timestamp)
                if ply_fpath is None:
                    continue
                lidar_timestamp = Path(ply_fpath).stem.split("_")[-1]
                lidar_timestamp = int(lidar_timestamp)
                labels = dl.get_labels_at_lidar_timestamp(
                    log_id, lidar_timestamp)
                if labels is None:
                    logging.info("\tLabels missing at t=%s", lidar_timestamp)
                    continue
                for label in labels:
                    obj_rec = json_label_dict_to_obj_record(label)
                    track_id = (obj_rec.track_id).replace("-", "")
                    ids.append(track_id)
            unique_ids = np.sort(np.unique(ids))
            id_dict = dict([(id, i) for i, id in enumerate(unique_ids)])

            # GENERATE PICKLE
            pickle_list = []
            for i, im_fpath in enumerate(cam_im_fpaths):

                cam_timestamp = Path(im_fpath).stem.split("_")[-1]
                cam_timestamp = int(cam_timestamp)

                # load PLY file path, e.g. 'PC_315978406032859416.ply'
                ply_fpath = dl.get_closest_lidar_fpath(log_id, cam_timestamp)
                if ply_fpath is None:
                    print(
                        f"LogID {log_id} Frame {i} : Lidar missing so we just put None"
                    )
                    img = imageio.imread(im_fpath).copy()
                    pl_entry = {
                        'center_camera_feed': np.array(img),
                        'obstacles': None
                    }
                else:
                    lidar_pts = load_ply(ply_fpath)
                    save_img_fpath = f"{save_dir}/{camera_name}_{cam_timestamp}.jpg"
                    # print (f"saving to {save_img_fpath}")
                    if Path(save_img_fpath).exists():
                        print("save_img_fpath DNE")
                        continue

                    city_to_egovehicle_se3 = dl.get_city_to_egovehicle_se3(
                        log_id, cam_timestamp)
                    if city_to_egovehicle_se3 is None:
                        print(
                            f'LogID {log_id} Frame {i} : Pose is missing so we just put None '
                        )
                        img = imageio.imread(im_fpath).copy()
                        pl_entry = {
                            'center_camera_feed': np.array(img),
                            'obstacles': None
                        }
                    else:
                        lidar_timestamp = Path(ply_fpath).stem.split("_")[-1]
                        lidar_timestamp = int(lidar_timestamp)
                        labels = dl.get_labels_at_lidar_timestamp(
                            log_id, lidar_timestamp)
                        if labels is None:
                            print(f"\tLabels missing at t={lidar_timestamp}")
                            logging.info("\tLabels missing at t=%s",
                                         lidar_timestamp)
                            continue

                        # Swap channel order as OpenCV expects it -- BGR not RGB
                        # must make a copy to make memory contiguous
                        img = imageio.imread(im_fpath).copy()
                        camera_config = get_calibration_config(
                            log_calib_data, camera_name)
                        planes = generate_frustum_planes(
                            camera_config.intrinsic.copy(), camera_name)

                        # print("We about to plot")

                        pl_entry = plot_img_2d_bboxes(
                            labels, planes, img, log_calib_data, camera_name,
                            cam_timestamp, lidar_timestamp, data_dir, log_id,
                            save_img_fpath, lidar_pts, id_dict, video)
                pickle_list.append(pl_entry)
                # print (f"Appending pl_entry #{i}")
                # print ("Finished plotting")
        assert len(pickle_list) == len(cam_im_fpaths)
        category_subdir = "2d_amodal_labels_100fr"

        if not Path(f"{experiment_prefix}_{category_subdir}").exists():
            os.makedirs(f"{experiment_prefix}_{category_subdir}")

        for cam_idx, camera_name in enumerate(RING_CAMERA_LIST +
                                              STEREO_CAMERA_LIST):
            # Write the cuboid video -- could also write w/ fps=20,30,40
            if camera_name != name_of_camera:
                continue
            if "stereo" in camera_name:
                fps = 5
            else:
                fps = 30
            img_wildcard = f"{save_dir}/{camera_name}_%*.jpg"
            output_fpath = f"{experiment_prefix}_{category_subdir}/{log_id}_{camera_name}_{fps}fps.mp4"
            pickle_output_fpath = f"{experiment_prefix}_{category_subdir}/{log_id}_{camera_name}_{fps}fps.pl"
            pickle.dump(pickle_list, open(pickle_output_fpath, "wb"))
            print(f'Total number of frames: {len(pickle_list)}')
            if video:
                write_nonsequential_idx_video(img_wildcard, output_fpath, fps)

            # print (img_wildcard, output_fpath,fps)


def main(args: Any):
    """Run the example."""
    log_ids = [log_id.strip() for log_id in args.log_ids.split(",")]
    dump_log_2d_bboxes_to_imgs(log_ids,
                               args.max_num_images_to_render * 9,
                               args.dataset_dir,
                               args.experiment_prefix,
                               video=args.video)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-num-images-to-render",
        default=5,
        type=int,
        help="number of images within which to render 3d cuboids")
    parser.add_argument("--dataset-dir",
                        type=str,
                        required=True,
                        help="path to the dataset folder")
    parser.add_argument(
        "--log-ids",
        type=str,
        required=True,
        help=("comma separated list of log ids, each log_id represents a log "
              "directory, e.g. found at "
              " {args.dataset-dir}/argoverse-tracking/train/{log_id} or "
              " {args.dataset-dir}/argoverse-tracking/sample/{log_id} or "),
    )
    parser.add_argument(
        "--experiment-prefix",
        default="output",
        type=str,
        help="results will be saved in a folder with this prefix for its name",
    )

    parser.add_argument(
        "--video",
        dest='video',
        action='store_true',
        help="video with bbox will be generated",
    )
    args = parser.parse_args()
    logger.info(args)

    if args.log_ids is None:
        logger.error("Please provide a comma seperated list of log ids")
        raise ValueError("Please provide a comma seperated list of log ids")

    main(args)
