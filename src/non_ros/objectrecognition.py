import tensorflow as tf
import numpy as np
import cv2

# Load a pre-trained MobileNetV2 model for object recognition
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Pre-process an image for the model
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# Start video capture
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    # Preprocess the frame and make a prediction
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)

    # Decode the predictions
    label = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
    
    # Display the label on the frame
    cv2.putText(frame, f"{label[1]}: {label[2]*100:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Object Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
