import numpy as np
import torch
import subprocess
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN


def main():
    # Post-inference clustering and visualization
    # TEST_DIR = "../datasets/images/val"
    TEST_DIR = "../datasets/images/test"
    TEST_LABEL_DIR = "../runs/crowd_yolov5s_detect/labels"
    RESULT_SAVE_DIR = "../image_results/20251016_v1"
    TRAINED_WEIGHT = "../model/weights/crowd_detector.pt"

    R = 80  # radius in pixels for density check
    MinPts = 3  # minimum points for core point
    ClusterSize_threshold = 5  # minimum cluster size for abnormal

    test_abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), TEST_DIR))

    print(f"\033[35m------ Running test  ------ \n R = {R}, MinPts = {MinPts}, ClusterSize_threshold = {ClusterSize_threshold}\033[0m")


    yolov5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolov5"))
    trained_weight = os.path.abspath(os.path.join(os.path.dirname(__file__), TRAINED_WEIGHT))

    print("Paths (absolute):")
    print("  TEST_DIR:", test_abs_dir)
    print("  TEST_LABEL_DIR:", os.path.abspath(os.path.join(os.path.dirname(__file__), TEST_LABEL_DIR)))
    print("  RESULT_SAVE_DIR:", os.path.abspath(os.path.join(os.path.dirname(__file__), RESULT_SAVE_DIR)))
    print("  TRAINED_WEIGHT:", trained_weight)

    cmd = [
        "python",
        os.path.join(yolov5_dir, "detect.py"),
        "--weights",
        trained_weight,
        "--source",
        test_abs_dir,
        "--img",
        "640",
        "--conf",
        "0.25",
        "--iou-thres",
        "0.15",
        "--save-csv",
        "--save-txt",
        "--save-conf",
        "--project",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs")),
        "--name",
        "crowd_yolov5s_detect",
        "--nosave",
    ]

    subprocess.run(cmd)

    print("\n\033[35m------ Inference completed, starting clustering and visualization ------\033[0m")

    result_save_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), RESULT_SAVE_DIR))
    os.makedirs(result_save_abs, exist_ok=True)

    def label_path_for_image(img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        # Resolve TEST_LABEL_DIR relative to this script file
        label_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), TEST_LABEL_DIR))
        lbl = os.path.join(label_dir, base + ".txt")
        return lbl if os.path.exists(lbl) else None

    def parse_yolo_label(lbl_path):
        if not lbl_path:
            return None
        boxes = []
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    boxes.append((cls, x, y, w, h))
        except Exception:
            raise ValueError("Failed to parse YOLO label file")
        return boxes

    # 生成三元组列表：(image_path, label_path_or_None, parsed_boxes_or_None)
    image_label_pairs = []
    

    if not os.path.isdir(test_abs_dir):
        raise FileNotFoundError(f"TEST_DIR not found: {test_abs_dir}")

    test_images = [
        os.path.join(test_abs_dir, f)
        for f in sorted(os.listdir(test_abs_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    for img in test_images:
        lp = label_path_for_image(img)
        parsed = parse_yolo_label(lp) if lp else None
        image_label_pairs.append((img, lp, parsed))

    # 快速检查：总数与有标签的数量，显示前10项（每项显示图片、标签路径、box数）
    total = len(image_label_pairs)
    with_labels = sum(1 for _, lp, _ in image_label_pairs if lp)
    print(f"total images: {total}, with label files: {with_labels}")

    for test_image_with_label in image_label_pairs:
        # test_image_with_label = image_label_pairs[210]

        img_path, _, parsed = test_image_with_label
        if parsed is None:
            print(f"  [skip] no label for {os.path.basename(img_path)}")
            continue

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        h, w = img.shape[:2]

        points = np.array([(x * w, y * h) for _, x, y, _, _ in parsed])


        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=R, min_samples=MinPts)
        labels = dbscan.fit_predict(points)

        # Group points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])

        # Identify abnormal clusters (exclude noise label -1)
        abnormal_clusters = [
            pts for label, pts in clusters.items() if label != -1 and len(pts) >= ClusterSize_threshold
        ]

        # Collect abnormal points
        abnormal_points = set()
        for cluster in abnormal_clusters:
            abnormal_points.update(tuple(pt) for pt in cluster)

        # Draw abnormal cluster circles (yellow, expanded by 5%)
        for cluster in abnormal_clusters:
            pts = np.array(cluster, dtype=np.float32)
            (x, y), radius = cv2.minEnclosingCircle(pts)
            radius *= 1.05  # expand by 5%
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 3)  # BGR: yellow

        # Draw points
        for pt in points:
            cx, cy = pt
            if tuple(pt) in abnormal_points:
                color = (0, 0, 255)  # red
            else:
                color = (0, 255, 0)  # green
            cv2.circle(img, (int(cx), int(cy)), 5, color, -1)  # filled

        # Save the result image (use imencode+tofile to support non-ASCII paths on Windows)
        result_path = os.path.join(result_save_abs, os.path.basename(img_path))
        ext = os.path.splitext(img_path)[1]
        cv2.imencode(ext, img)[1].tofile(result_path)
        print(f"Saved result to {result_path}")

        # Display the image (convert BGR to RGB for matplotlib)
        # plt.figure(dpi = 300)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()


if __name__ == "__main__":
    main()
