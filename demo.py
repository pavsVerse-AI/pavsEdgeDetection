import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Check if it's connected and permissions are allowed.")
    exit()

print("Webcam started! Press 'q' to quit. Try waving a hand or object in front of the camera.")

# --- Create configurable windows ---
cv2.namedWindow('Raw Feed (Original)', cv2.WINDOW_NORMAL)
cv2.namedWindow('Edges (Canny Output)', cv2.WINDOW_NORMAL)
cv2.namedWindow('Outlined Objects (Contours)', cv2.WINDOW_NORMAL)

# --- Resize and move them ---
cv2.resizeWindow('Raw Feed (Original)', 480, 360)
cv2.resizeWindow('Edges (Canny Output)', 480, 360)
cv2.resizeWindow('Outlined Objects (Contours)', 480, 360)

cv2.moveWindow('Raw Feed (Original)', 0, 0)
cv2.moveWindow('Edges (Canny Output)', 500, 0)
cv2.moveWindow('Outlined Objects (Contours)', 1000, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    outlined = frame.copy()
    cv2.drawContours(outlined, contours, -1, (0, 255, 0), 2)

    # Add text
    cv2.putText(outlined, f'Edge Detection Active! Found {len(contours)} shapes',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show all frames
    cv2.imshow('Raw Feed (Original)', frame)
    cv2.imshow('Edges (Canny Output)', edges)
    cv2.imshow('Outlined Objects (Contours)', outlined)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("All done! Your edges were detected like a pro. 🎉")
