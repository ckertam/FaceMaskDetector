import cv2
from mtcnn import MTCNN
import tensorflow as tf

# Load the trained mask detection model
model = tf.keras.models.load_model('mask_detector2.model')

# Initialize MTCNN face detector
detector = MTCNN()

# Define image height and width
Your_Image_Height, Your_Image_Width = 150, 150  # Replace with your values

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Break the loop with 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Detect faces in the frame using MTCNN
    faces = detector.detect_faces(frame)
    for face in faces:
        # Get the bounding box of the face
        x, y, w, h = face['box']

        # Extract face ROI (Region of Interest)
        face_roi = frame[y:y+h, x:x+w]

        # Resize to match the input size of the model
        resized_face = cv2.resize(face_roi, (Your_Image_Height, Your_Image_Width))

        # Normalize the face ROI
        normalized_face = resized_face / 255.0

        # Make prediction
        prediction = model.predict(tf.expand_dims(normalized_face, axis=0))

        # Determine label and color
        if prediction[0][0] > 0.7:
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

    # Wait for 100 ms (or any other duration) before moving to the next frame
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
