import cv2
import numpy as np
import pandas as pd

# Configuration paths
config_path = 'cfg/yolov4_stars_LCO.cfg'
weights_path = 'backup/yolov4_stars_LCO_best.weights'
input_size = (1056, 1056)
conf_thresh = 0.1  # Confidence threshold
nms_thresh = 0.1  # Non-maximum suppression threshold

# Setup YOLO model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Read test file paths
test_file = "LCO/sets/testLCO.txt"
with open(test_file) as f:
    test_paths = [line.strip().replace("/1.png", "") for line in f]

all_results = []

for path in test_paths:
    img_path = path + '/1.png'
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image at {img_path} could not be loaded.")
        continue

    # Load ground truths
    ground_truths = []
    try:
        with open(f"{path}/1.txt", "r") as f:
            ground_truths = [[3000 * float(line.split()[1]), 2000 * float(line.split()[2])] for line in f]
    except FileNotFoundError:
        print(f"Ground truth file at {path}/1.txt could not be found.")
        continue

    # Create blob from image
    blob = cv2.dnn.blobFromImage(img, 1/255.0, input_size, [0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)

    # Get detections
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    h, w = img.shape[:2]

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]  # Extract class scores
            class_id = np.argmax(scores)  # Find the index of the highest score
            confidence = scores[class_id]  # Get the highest confidence score

            if confidence > conf_thresh:
                box = detection[0:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype('int')
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    if len(indices) == 0:
        print(f"No detections for image {img_path}")
        continue

    detected_centers = set()
    matched_ground_truths = set()

    for i in indices.flatten():
        box = boxes[i]
        x, y, width, height = box
        center_x = x + width / 2
        center_y = y + height / 2
        center = (center_x, center_y)
        confidence = confidences[i]
        match_found = False

        for j, ground_truth in enumerate(ground_truths):
            if abs(ground_truth[0] - center[0]) <= 10 and abs(ground_truth[1] - center[1]) <= 10:
                all_results.append([
                    img_path, confidence, False, ground_truth[0], ground_truth[1], center[0], center[1]
                ])
                matched_ground_truths.add(j)
                match_found = True
                break

        if not match_found:
            all_results.append([
                img_path, confidence, True, "falsepos", "falsepos", center[0], center[1]
            ])
            detected_centers.add((center[0], center[1]))

    for j, ground_truth in enumerate(ground_truths):
        if j not in matched_ground_truths:
            all_results.append([
                img_path, "not_found", True, ground_truth[0], ground_truth[1], "falseneg", "falseneg"
            ])

# Save results to CSV
pd.DataFrame(
    all_results,
    columns=['img', 'confidence', 'false pos', 'true x', 'true y', 'pred x', 'pred y']
).to_csv('/mnt/annex/rachel/YOLO_data/darknet/performance.csv', index=False)
s
