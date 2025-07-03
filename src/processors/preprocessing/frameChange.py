import cv2
import time

def is_significant_change(frame1, frame2, region_threshold, region_size=(50, 50)):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    diff = cv2.absdiff(gray1, gray2)
    height, width = gray1.shape
    upper_diff = diff[:height // 2, :]
    lower_diff = diff[height // 2:, :]

    def check_regions(diff_region):
        total_regions = (diff_region.shape[0] // region_size[1]) * (diff_region.shape[1] // region_size[0])
        significant_regions = 0

        for y in range(0, diff_region.shape[0], region_size[1]):
            for x in range(0, diff_region.shape[1], region_size[0]):
                region = diff_region[y:y + region_size[1], x:x + region_size[0]]
                if cv2.countNonZero(region) > region_threshold:
                    significant_regions += 1

        return significant_regions > (total_regions * 0.2)

    upper_significant = check_regions(upper_diff)
    lower_significant = check_regions(lower_diff)

    return upper_significant, lower_significant, diff

def highlight_changes(diff, frame):
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:

        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
        if x_min < float('inf') and y_min < float('inf'):
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    return frame

def process_video(input_video_path, output_video_path, region_threshold, region_size=(50, 50)):
    start_time = time.time()

    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    out.write(prev_frame)

    total_frames = 1
    significant_change_frames = 0
    skipped_frames = 0

    frame_processing_times = []

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        upper_significant, lower_significant, diff = is_significant_change(prev_frame, curr_frame, region_threshold, region_size)
        frame_end_time = time.time()

        frame_processing_times.append(frame_end_time - frame_start_time)

        if upper_significant or lower_significant:
            if upper_significant:
                print(f"Significant change detected in upper region of frame {total_frames}")
            if lower_significant:
                print(f"Significant change detected in lower region of frame {total_frames}")

            out.write(curr_frame)
            significant_change_frames += 1
        else:
            out.write(prev_frame)
            skipped_frames += 1

        diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        highlighted_frame = highlight_changes(diff_thresh, curr_frame.copy())

        diff_resized = cv2.resize(highlighted_frame, (640, 360))
        cv2.imshow('Frame Difference', diff_resized)
        cv2.resizeWindow('Frame Difference', 640, 360)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        prev_frame = curr_frame
        total_frames += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    processing_time = end_time - start_time

    avg_frame_processing_time = sum(frame_processing_times) / len(frame_processing_times)

    print(f"Total frames: {total_frames}")
    print(f"Significant change frames: {significant_change_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Total processing time: {processing_time} seconds")
    print(f"Average time per frame pair: {avg_frame_processing_time} seconds")


input_video = '2.mp4'
output_video = 'output_video.mp4'
region_threshold = 1500  #threshold for regions
region_size = (50, 50)  # Define the size of each region

process_video(input_video, output_video, region_threshold, region_size)
