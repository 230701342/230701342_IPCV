import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

    video_path = 'traffic.mp4' 
    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        print("Please ensure the video file is in the same directory as the script.")
        return

 
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    

    kernel = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)

    vehicle_counts_history = []
    frame_count = 0

    print("Processing video... Press 'q' to exit early.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
       
        frame = cv2.resize(frame, (800, 600))
        height, width, _ = frame.shape
        mid_line = width // 2 

        roi_y1, roi_y2 = 250, 580
        roi_x1, roi_x2 = 20, 780
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

       
        fg_mask = bg_subtractor.apply(roi)

       
        _, fg_mask = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)
        
        # Opening to remove noise, Closing to join vehicle parts
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)

        # 5. CONTOUR DETECTION
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        left_lane_count = 0
        right_lane_count = 0

        for cnt in contours:
            # Significant reduction in area threshold to 150 for small vehicles
            area = cv2.contourArea(cnt)
            if area > 150: 
                # Get bounding box coordinates relative to ROI
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter by Aspect Ratio and Minimum Dimensions (Lowered for sensitivity)
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 4.0 and w > 20 and h > 20:
                    # Draw Bounding Box on original frame (offset by ROI coordinates)
                    cv2.rectangle(frame, (x + roi_x1, y + roi_y1), (x + w + roi_x1, y + h + roi_y1), (0, 255, 0), 2)
                    
                    # Add Label
                    cv2.putText(frame, "Vehicle", (x + roi_x1, y + roi_y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Identify Lane based on X-coordinate of centroid (relative to full frame)
                    centroid_x = x + roi_x1 + (w // 2)
                    if centroid_x < mid_line:
                        left_lane_count += 1
                    else:
                        right_lane_count += 1

        total_vehicles = left_lane_count + right_lane_count
        vehicle_counts_history.append(total_vehicles)

        # 6. TRAFFIC DENSITY CLASSIFICATION
        if total_vehicles <= 10:
            density = "LOW"
            color = (0, 255, 0)   # Green
        elif total_vehicles <= 25:
            density = "MEDIUM"
            color = (0, 255, 255) # Yellow
        else:
            density = "HIGH"
            color = (0, 0, 255)   # Red

        # 7. REAL-TIME OVERLAY
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 150), (0, 0, 0), -1)
        # Draw ROI boundary (optional visualization)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Draw the lane divider
        cv2.line(frame, (mid_line, roi_y1), (mid_line, roi_y2), (255, 255, 255), 1)

        # Put Text
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Left Lane: {left_lane_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        cv2.putText(frame, f"Right Lane: {right_lane_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        cv2.putText(frame, f"Total: {total_vehicles}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Density Status
        cv2.putText(frame, f"STATUS: {density}", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # 8. DISPLAY
        cv2.imshow("Smart Traffic Monitoring System", frame)
        # cv2.imshow("Foreground Mask", fg_mask) # Uncomment to see the detection mask

        # Break loop on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 9. CLEANUP
    cap.release()
    cv2.destroyAllWindows()

    # 10. PLOT GRAPH
    plt.figure(figsize=(10, 5))
    plt.plot(vehicle_counts_history, color='blue', linewidth=2)
    plt.title('Vehicle Count Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Vehicles')
    plt.grid(True)
    plt.savefig('traffic_density_graph.png')
    print("Processing complete. Graph saved as 'traffic_density_graph.png'.")
    plt.show()

if __name__ == "__main__":
    main()
