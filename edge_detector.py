import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Check if it's connected and permissions are allowed.")
    exit()

print("Webcam started! Press 'q' to quit. Try waving a hand or object in front of the camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = frame.copy()
    cv2.drawContours(outlined, contours, -1, (0, 255, 0), 2)
    cv2.putText(outlined, 'Edge Detection Active! Found {} shapes'.format(len(contours)), 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow('Raw Feed (Original)', frame)
    cv2.imshow('Edges (Canny Output)', edges)
    cv2.imshow('Outlined Objects (Contours)', outlined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("All done! Your edges were detected like a pro. 🎉")