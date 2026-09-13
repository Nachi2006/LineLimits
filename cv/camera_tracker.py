import cv2
import numpy as np
import json
import sys

def main():
    # ==========================================
    # Step 1: Setup and Loading
    # ==========================================
    # Load the static master map (grayscale)
    master_map = cv2.imread('master_map.jpg', cv2.IMREAD_GRAYSCALE)
    if master_map is None:
        print("Error: Could not load master_map.jpg. Please ensure the file exists.")
        sys.exit(1)

    # Load the video file
    cap = cv2.VideoCapture('video.mp4')
    if not cap.isOpened():
        print("Error: Could not open video.mp4. Please ensure the file exists.")
        sys.exit(1)

    # Load the JSON polygon
    # Expected JSON format: [[x1, y1], [x2, y2], ..., [xN, yN]]
    try:
        with open('polygon.json', 'r') as f:
            polygon_data = json.load(f)
        # Convert to NumPy array of shape (N, 1, 2) of type float32 for OpenCV
        polygon_pts = np.array(polygon_data, dtype=np.float32).reshape(-1, 1, 2)
    except FileNotFoundError:
        print("Error: Could not find polygon.json. Creating a dummy polygon for testing.")
        # Dummy polygon: A simple rectangle if JSON is missing
        h, w = master_map.shape
        polygon_pts = np.array([
            [w//4, h//4],
            [3*w//4, h//4],
            [3*w//4, 3*h//4],
            [w//4, 3*h//4]
        ], dtype=np.float32).reshape(-1, 1, 2)
    except Exception as e:
        print(f"Error parsing polygon.json: {e}")
        sys.exit(1)

    # ==========================================
    # Step 2: Feature Extraction (Master Map)
    # ==========================================
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for the master map
    kp_map, des_map = sift.detectAndCompute(master_map, None)
    
    if des_map is None or len(des_map) < 10:
        print("Error: Not enough features found in master_map.jpg.")
        sys.exit(1)

    # Initialize FLANN based matcher
    # FLANN parameters for SIFT (algorithm = 1 is FLANN_INDEX_KDTREE)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    print("Starting video loop. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        # Convert live frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ==========================================
        # Step 2 (Cont.): Feature Extraction (Live Frame)
        # ==========================================
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

        if des_frame is None or len(des_frame) < 10:
            cv2.imshow('Dynamic Camera Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ==========================================
        # Step 3: Feature Matching
        # ==========================================
        matches = flann.knnMatch(des_frame, des_map, k=2)

        # Apply Lowe's Ratio Test to keep good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # ==========================================
        # Step 4: Homography Matrix Calculation
        # ==========================================
        if len(good_matches) > 10:
            # Extract (x, y) coordinates for the good matches
            # queryIdx refers to des_frame (live frame), trainIdx refers to des_map (master map)
            src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate Homography from Live Frame -> Master Map
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # ==========================================
                # Step 5: Reverse Projection (The Visual Test)
                # ==========================================
                try:
                    # Calculate inverse Homography (Master Map -> Live Frame)
                    H_inv = np.linalg.inv(H)
                    
                    # Transform the polygon from Master Map perspective to Live Frame perspective
                    transformed_polygon = cv2.perspectiveTransform(polygon_pts, H_inv)
                    
                    # Convert to int32 for polylines drawing
                    transformed_polygon_int = np.int32(transformed_polygon)
                    
                    # Draw the polygon on the live color frame
                    # isClosed=True, color=(0, 255, 0) for green, thickness=3
                    frame = cv2.polylines(frame, [transformed_polygon_int], True, (0, 255, 0), 3)
                    
                    # Optional: Add text indicating successful track
                    cv2.putText(frame, f"Tracking Active (Matches: {len(good_matches)})", 
                                (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                except np.linalg.LinAlgError:
                    # In case the Homography matrix is singular/non-invertible
                    cv2.putText(frame, "Math Error: Singular Matrix", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Tracking Lost: H-Matrix failed", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Tracking Lost: Not enough matches ({len(good_matches)}/10)", 
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ==========================================
        # Step 6: Display
        # ==========================================
        cv2.imshow('Dynamic Camera Tracking', frame)

        # Allow video playback and 'q' key exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
