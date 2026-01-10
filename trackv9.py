import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import platform
import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from time import time
import pickle


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov9') not in sys.path:
    sys.path.append(str(ROOT / 'yolov9'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from yolov9.models.experimental import attempt_load
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.dataloaders import LoadImages, LoadStreams, LoadScreenshots
from yolov9.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov9.utils.torch_utils import select_device, time_sync, smart_inference_mode
from yolov9.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



@smart_inference_mode()
def run(
        source='0',
        data = ROOT / 'data/coco.yaml',  # data.yaml path
        yolo_weights=WEIGHTS / 'yolo.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        gallery='gallery.pkl',  # path to gallery
        save_log=False, # Save ID logs
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')

    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
        yolo_weights = Path(yolo_weights[0])
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir = Path(save_dir)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize Log File
    log_path = save_dir / 'attendance_log.csv'
    if save_log:
        with open(log_path, 'w') as f:
            f.write('Frame,Timestamp,TrackID,Name,Confidence,Dist\n')
        print(f"Logging attendance to {log_path}")
    
    log_count = 0

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer,txt_path = [None] * bs, [None] * bs, [None] * bs
    
    
    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(bs):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * bs
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Load gallery once
    gallery_dict = {}
    gallery_path = gallery
    if not os.path.exists(gallery_path):
        gallery_path = ROOT / gallery
    
    if os.path.exists(gallery_path):
         with open(gallery_path, "rb") as f:
            gallery_dict = pickle.load(f)
            print(f"Loaded gallery from {gallery_path} with {len(gallery_dict)} identities: {list(gallery_dict.keys())}")
    else:
        print(f"Gallery not found at {gallery_path}, running without identification.")

    # OPTIMIZATION: Vectorize Gallery for fast matrix multiplication
    gallery_matrix = []
    gallery_labels = [] # Parallel list to store names corresponding to rows
    if gallery_dict:
        print("Vectorizing gallery for performance...")
        for name, feats in gallery_dict.items():
             if not isinstance(feats, list):
                 feats = [feats]
             for f in feats:
                 # Ensure shape is correct (sometimes could be 1, 512)
                 f = f.reshape(-1)
                 gallery_matrix.append(f)
                 gallery_labels.append(name)
        
        if gallery_matrix:
            gallery_matrix = np.array(gallery_matrix) # Shape: (N_looks, 512)
            print(f"Gallery Matrix Shape: {gallery_matrix.shape}")
        else:
            print("Warning: Gallery dictionary was empty or invalid.")
    
    # OPTIMIZATION: Cache for identified tracks to avoid re-computing every frame
    # Format: {track_id: {'name': name, 'last_check': frame_idx}}
    id_cache = {}
    
    # Store distances for logging (TrackID -> Dist)
    dist_map = {}



    # Run tracking
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt,sdt = 0, [], (Profile(), Profile(), Profile(), Profile()),[0.0, 0.0, 0.0, 0.0]
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        # s = ''
        t1 = time_sync()
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        t2 = time_sync()
        sdt[0] += t2 - t1

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            pred = pred[0]
        t3 = time_sync()
        sdt[1] += t3 - t2

        # Apply NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        sdt[2] += time_sync() - t3
        
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                # txt_file_name = p.name
                txt_file_name = p.stem + f'_{i}' # Unique text file name
                # save_path = str(save_dir / p.name) + str(i)  # im.jpg, vid.mp4, ...
                save_path = str(save_dir / p.stem) + f'_{i}'  # Unique video file name

            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
                
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                sdt[3] += t5 - t4

                # ID Assignment loop
                id_map = {}
                
                # Use pre-loaded gallery_dict
                if gallery_dict and len(gallery_matrix) > 0:
                     # Iterate over tracks and match against gallery
                     for t in strongsort_list[i].tracker.tracks:
                         if not t.is_confirmed() or t.time_since_update > 1:
                             continue
                         
                         # OPTIMIZATION: Persistent Caching Logic
                         # Check if we already know this track ID from previous frames
                         cached_name = id_cache.get(t.track_id)
                         
                         should_run_matching = True
                         if cached_name:
                             # We know this person. 
                             # Optimization: Only re-verify every 30 frames to save compute.
                             if frame_idx % 30 != 0:
                                 should_run_matching = False
                                 # Trust the cache for this frame
                                 id_map[t.track_id] = cached_name
                         
                         if should_run_matching and len(t.features) > 0:
                             feat = t.features[-1]
                             feat = feat / np.linalg.norm(feat) # Ensure normalized
                             feat = feat.reshape(-1) # Ensure (512,)
                             
                             # OPTIMIZATION: Vectorized Cosine Distance
                             # Dist = 1 - (Gallery . Track)
                             # Shape: (N,) = (N, 512) . (512,)
                             scores = np.dot(gallery_matrix, feat)
                             dists = 1.0 - scores
                             
                             min_idx = np.argmin(dists)
                             min_dist = dists[min_idx]
                             
                             # Threshold check (kept at 0.1)
                             if min_dist < 0.1:
                                 best_name = gallery_labels[min_idx]
                                 id_map[t.track_id] = best_name
                                 id_cache[t.track_id] = best_name # Update Persistent Cache
                                 dist_map[t.track_id] = min_dist # Store for logging
                                 
                                 if tuple(dists).index(min_dist) == 0: 
                                      pass 
                             else:
                                 # If re-verification failed, remove from cache (person left or occlusion)
                                 if t.track_id in id_cache:
                                     del id_cache[t.track_id]


                
                # OPTIMIZATION / FEATURE: Logging
                if save_log:
                    # Calculate timestamp
                    # current_time_sec = frame_idx * vid_stride / 30.0 # Approximation if FPS unknown
                    # Better to use dataset properties if available, but for now generic:
                    timestamp = frame_idx * vid_stride / 30.0 
                    
                    for t in strongsort_list[i].tracker.tracks:
                        if t.is_confirmed() and t.time_since_update <= 1:
                             name = id_map.get(t.track_id, "Unknown")
                             
                             # Retrieve real values
                             conf_val = getattr(t, 'conf', -1)
                             if isinstance(conf_val, torch.Tensor):
                                 conf_val = conf_val.item()
                             
                             dist_val = dist_map.get(t.track_id, -1)
                             if dist_val != -1 and isinstance(dist_val, (np.float32, np.float64)):
                                 dist_val = f"{dist_val:.4f}"

                             with open(log_path, 'a') as f:
                                 # Frame, Timestamp, TrackID, Name, Confidence, Dist
                                 f.write(f"{frame_idx},{timestamp:.2f},{t.track_id},{name},{conf_val:.2f},{dist_val}\n")
                             log_count += 1


                # Write results
                for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                    xyxy = output[0:4]
                    id = output[4]
                    cls = output[5]
                # for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # line = (id , cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        line = ( int(p.stem), frame_idx, id , cls, *xywh, conf) if save_conf else ( p.stem, frame_idx, cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as file:
                            file.write(('%g ' * len(line) + '\n') % line)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # Custom Identification
                        identity = id_map.get(int(id), str(int(id)))
                        label = None if hide_labels else (names[c] if hide_conf else f' { identity } {names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)




                # Draw boxes
                # if len(outputs[i]) > 0:
                #     for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                #         bboxes = output[0:4]
                #         id = output[4]
                #         cls = output[5]

                #         if save_txt:
                #             # to MOT format
                #             bbox_left = output[0]
                #             bbox_top = output[1]
                #             bbox_w = output[2] - output[0]
                #             bbox_h = output[3] - output[1]
                #             # format video_name frame id xmin ymin width height score class 
                #             with open(txt_path + '.txt', 'a') as file:
                #                 file.write(f'{p.stem} {frame_idx} {id} {bbox_left} {bbox_top} {bbox_w} {bbox_h} {conf:.2f} {cls}\n')

                #         if save_img or save_crop or view_img:  # Add bbox to image
                #             c = int(cls)  # integer class
                #             id = int(id)  # integer id
                #             
                #             # CUSTOM: Use name if identified
                #             identity = id_map.get(id, str(id))
                #             
                #             label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {identity} {conf:.2f}')
                #             plot_one_box(bboxes, im0, label=label, color=colors[int(cls)], line_thickness=2)
                #             if save_crop:
                #                 txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                #                 save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                print('No detections')

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print time (inference-only)
    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    # Print results
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape, %.1fms StrongSORT' % tuple(1E3 * x / seen for x in sdt))
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    
    if save_log:
        LOGGER.info(f"Attendance Log saved to {log_path} ({log_count} entries)")
        
        # IMPROVISATION: Generate Summary Report
        try:
             import pandas as pd
             if log_count > 0:
                 summary_path = save_dir / 'attendance_summary.csv'
                 df = pd.read_csv(log_path)
                 
                 # Group by Name to handle re-entries across different TrackIDs if identified
                 # Or Group by TrackID first? User wants "1 testperson enter left enter left"
                 # Strategy: Filter valid Names (not Unknown if possible, or include Unknowns by TrackID)
                 
                 session_rows = []
                 
                 # FEATURE: Filter out 'Unknown' from summary as requested
                 df_filtered = df[~df['Name'].str.contains('Unknown', case=False, na=False)]
                 
                 if df_filtered.empty:
                      LOGGER.info("No identified people found for summary.")
                 else:
                     # Process each unique Name
                     for name, group in df_filtered.groupby('Name'):
                         group = group.sort_values('Timestamp')
                         timestamps = group['Timestamp'].values
                         
                         if len(timestamps) == 0:
                             continue
                             
                         # Detect separate sessions (Gap > 3 seconds)
                         # Reduced to 3s to be more sensitive to leaving/re-entering
                         SESSION_GAP_THRESHOLD = 3.0 
                         
                         # Identify indices where the gap is large
                         diffs = np.diff(timestamps)
                         split_indices = np.where(diffs > SESSION_GAP_THRESHOLD)[0] + 1
                         
                         # Split timestamps into sessions
                         sessions = np.split(timestamps, split_indices)
                         
                         session_count = 0
                         for session_times in sessions:
                             if len(session_times) == 0:
                                 continue
                             start_t = session_times[0]
                             end_t = session_times[-1]
                             duration = end_t - start_t
                             
                             # NOISE FILTER: Ignore sessions < 1.0 second
                             if duration < 1.0:
                                 continue
                             
                             session_count += 1
                             session_rows.append({
                                 'Name': name,
                                 'Session_ID': session_count,
                                 'First_Seen': f"{start_t:.2f}",
                                 'Last_Seen': f"{end_t:.2f}",
                                 'Duration_Sec': f"{duration:.2f}",
                                 'Frame_Count': len(session_times)
                             })
                     
                     if session_rows:
                         summary_df = pd.DataFrame(session_rows)
                         summary_df = summary_df.sort_values(['Name', 'First_Seen'])
                         summary_df.to_csv(summary_path, index=False)
                         LOGGER.info(f"Attendance SUMMARY (Sessions) saved to {summary_path}")
                     else:
                        LOGGER.info("No valid sessions (>1s) found for summary.")


        except ImportError:
             LOGGER.info("Pandas not installed, skipping summary report.")
        except Exception as e:
             LOGGER.info(f"Could not generate summary: {e}")

    if update:
        strip_optimizer(yolo_weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov9.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--gallery', type=str, default='gallery.pkl', help='path to gallery.pkl for person identification')
    parser.add_argument('--save-log', action='store_true', help='save attendance log to csv')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    # check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
