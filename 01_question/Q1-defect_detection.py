# imports
import os
import sys
import cv2
import numpy as np

# global variable
defect_label_map = {
    0: "cut",
    1: "flash",
}


def get_circle_contour(cnt):
    """Form a circular contour from the input contour using convexity defects."""
    # get convexity defects
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_indices)
    # remove unnecessary defective points
    defective_indices = np.where(defects[:, :, 3] > 256)[0]
    if len(defective_indices) > 1:
        remove_indices = []
        for i in range(len(defective_indices)):
            if abs(defective_indices[i] - defective_indices[i - 1]) > 5:
                remove_indices.append(defective_indices[i])
            elif len(remove_indices) > 0:
                remove_indices.pop()
    else:
        remove_indices = defective_indices
    defects = np.delete(defects, remove_indices, axis=0)
    # ensure a continuous circular path for the contour
    continuous_path = True
    circle_cnt = []
    for [defect] in defects:
        far = cnt[defect[2]]
        if defect[3] > 256:
            continuous_path = not continuous_path
        if continuous_path:
            circle_cnt.append(far)
    return np.array(circle_cnt)


def get_defects(diff, base_cnt, ring_type):
    """Identify defects (cuts or flashes) within the given difference image."""
    base_cnt_area = cv2.contourArea(base_cnt)
    # detect contours in the difference
    cnts, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cut_or_flash_flags = []
    cut_or_flash_bboxes = []
    for c in cnts:
        # contours with less 0.01% of base contour area are ignored
        if cv2.contourArea(c) < (base_cnt_area * 0.0001):
            continue
        # check center point of the contour lies inside outside of the base contour
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        inside_flag = int(
            cv2.pointPolygonTest(base_cnt, (cx, cy), measureDist=False) >= 0
        )
        # decide defect type based on the ring type (inner or outer)
        if ring_type == "inner":
            cut_or_flash = inside_flag
        elif ring_type == "outer":
            cut_or_flash = abs(inside_flag - 1)
        cut_or_flash_flags.append(cut_or_flash)
        # get bounding box for the defect
        x0, y0, w, h = cv2.boundingRect(c)
        cut_or_flash_bboxes.append([x0, y0, x0 + w, y0 + h])
    return cut_or_flash_flags, cut_or_flash_bboxes


def save_output_image(image, image_path, defect_labels, defect_bboxes):
    """Save the output image with labeled defects."""
    output_image = image.copy()
    output_path = (
        image_path[: image_path.rfind(".")]
        + "_detections"
        + image_path[image_path.rfind(".") :]
    )
    for defect_label, defect_bbox in zip(defect_labels, defect_bboxes):
        xmin, ymin, xmax, ymax = defect_bbox
        label = defect_label_map[defect_label]
        # Uncomment the lines below to draw bounding boxes around defects and circles at center
        # cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        # cv2.circle(output_image, ((xmin+xmax)//2, (ymin+ymax)//2), 5, (0, 0, 255), -1)
        cv2.putText(
            output_image,
            label,
            (xmin, (ymin + ymax) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            5,
        )
    cv2.imwrite(output_path, output_image)
    print(f"The output image with defects is saved in '{output_path}' path.")


def print_results(input_image, overall_defect_labels):
    """Print the overall defects detected in the given input image."""
    if len(overall_defect_labels) == 0:
        print_out = f"GOOD. The given image '{input_image}' has no defects."
    elif sum(overall_defect_labels) == 0:
        print_out = f"CUT. The given image '{input_image}' has the defect cut."
    elif sum(overall_defect_labels) == len(overall_defect_labels):
        print_out = f"FLASH. The given image '{input_image}' has the defect flash."
    else:
        print_out = f"The given image '{input_image}' has both the defects cut & flash."
    print(print_out)


def main(input_image):
    """Main function to process the input image and detect defects."""
    try:
        image = cv2.imread(input_image)
    except Exception as e:
        print(f"Cannot process the file '{input_image}'.")
        print(f"Exception: {e}")
        sys.exit()
    height, width, _ = image.shape
    total_area = height * width
    # convert color (BGR) image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # convert grayscale image to binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    # detect contours in the binary image
    contours, [hierarchies] = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    ring_type = None
    overall_defect_labels = []
    overall_defect_bboxes = []
    for cnt, hier in zip(contours, hierarchies):
        cnt_area = cv2.contourArea(cnt)
        # contours with less 0.1% of total image area are ignored
        if cnt_area < (0.001 * total_area):
            continue
        # based contour hierarchy choose ring type (inner or outer)
        ring_type = "outer" if hier[-1] == -1 else "inner"
        # get the circular contour for the detected contour
        circle_cnt = get_circle_contour(cnt)
        # create mask for circular contour
        circle_mask = cv2.drawContours(np.zeros_like(thresh), [circle_cnt], -1, 255, -1)
        # create mask for detected contour
        contour_mask = cv2.drawContours(np.zeros_like(thresh), [cnt], -1, 255, -1)
        # get the differences between the masks
        diff = cv2.absdiff(contour_mask, circle_mask)
        # reduce the noise in the difference
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
        # if there is no difference in this contour, this a perfect circle contour and ignore it
        if np.count_nonzero(diff) == 0:
            continue
        # if there are differences in this contour, get those defects' label and bbox
        defect_labels, defect_bboxes = get_defects(diff, circle_cnt, ring_type)
        overall_defect_labels.extend(defect_labels)
        overall_defect_bboxes.extend(defect_bboxes)

    print_results(input_image, overall_defect_labels)
    # if there are defects, save the results
    if len(overall_defect_labels) > 0:
        save_output_image(
            image, input_image,
            overall_defect_labels,
            overall_defect_bboxes
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
        if not os.path.exists(input_image):
            print(f"No such image file: {input_image}")
            sys.exit()
        main(input_image)
    else:
        print("Give an image filepath as argument.")
        print("For example: `python3 image.jpg`")
