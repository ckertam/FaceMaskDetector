import cv2
import tensorflow as tf
import numpy as np

# Load the trained mask detection model
model = tf.keras.models.load_model('mask_detector.model')

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define image height and width (set these to the dimensions you used during training)
Your_Image_Height, Your_Image_Width = 150, 150  # Replace with your values

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract face ROI (Region of Interest)
        face_roi = frame[y:y+h, x:x+w]

        # Resize to match the input size of the model
        resized_face = cv2.resize(face_roi, (Your_Image_Height, Your_Image_Width))

        # Normalize the face ROI
        normalized_face = resized_face / 255.0

        # Make prediction
        prediction = model.predict(tf.expand_dims(normalized_face, axis=0))

        # Determine label and color
        if prediction[0][0] > 0.5:
            label = "No Mask {:.2f}%".format((1 - prediction[0][0]) * 100)
            color = (0, 0, 255)  # Red for no mask
        else:
            label = "Mask {:.2f}%".format(prediction[0][0] * 100)
            color = (0, 255, 0)  # Green for mask

        # Draw rectangle around the face and put label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('Mask Detection', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
