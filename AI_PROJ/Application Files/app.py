from flask import Flask, render_template ,request , send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load the YOLOv3 model
net = cv2.dnn.readNetFromDarknet("C:\\Users\\Vishnu\\Downloads\\yolov3_custom_testing.cfg", r"C:\\Users\\Vishnu\\Downloads\\yolov3_custom_4000.weights")

# Load the class labels
classes = ['Drowning', 'Swimming']
@app.route("/")
def hello():
    return render_template('index2.html')

@app.route('/predict',methods= ['GET','POST'])
def index():
    # Load the input image
    image_file = request.files['image']

    # Save the uploaded image
    input_image_path = os.path.join(BASE_DIR, "uploads", "input_image.jpg")
    image_file.save(input_image_path)

    # Load the input image
    image = cv2.imread(input_image_path)

    # Preprocess the image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Forward pass through the network
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Process the outputs
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Scale the bounding box coordinates
                box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (center_x, center_y, width, height) = box.astype("int")

                # Calculate the top-left corner coordinates
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Draw bounding boxes and labels
    for i in indices.flatten():
        (x, y, w, h) = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image
    output_image_path = "uploads/output_image.jpg"
    cv2.imwrite(output_image_path, image)

    # Pass the image path to the template
    return render_template('output2.html', image_path=output_image_path)

@app.route('/predict_video', methods=['POST'])
def predict_video():
    # Load the input video
    video_file = request.files['video']
    video_path = os.path.join(BASE_DIR, "uploads", "input_video.mp4")
    video_file.save(video_path)

    # Set the confidence threshold and NMS threshold
    confidence_threshold = 0.5
    nms_threshold = 0.3

    # Load the input video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define output folder

    frame_counter = 0

    while cap.isOpened():
        # Read the frame
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Set the input blob for the network
        net.setInput(blob)

        # Forward pass through the network
        output_layers_names = net.getUnconnectedOutLayersNames()
        layer_outputs = net.forward(output_layers_names)

        # Process the outputs
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    # Scale the bounding box coordinates
                    box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (center_x, center_y, width, height) = box.astype("int")

                    # Calculate the top-left corner coordinates
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y, w, h) = boxes[i]
                label = classes[class_ids[i]]
                confidence = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the processed frame as an image
            output_folder = "uploads/"
            output_filename = f"{output_folder}frame_{frame_counter}.jpg"
            cv2.imwrite(output_filename, frame)
            frame_counter += 1

        # Display the processed frame
        cv2.imshow("Output", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Save the output video frame
    output_frame_path = "uploads/output_frame.jpg"
    cv2.imwrite(output_frame_path, frame)


    # Pass the frame path to the template
    return render_template('output3video.html', video_path=output_frame_path, frame_counter=frame_counter)

@app.route('/retur',methods= ['GET','POST'])
def retur():
    return render_template('index2.html')

@app.route('/returvideo',methods= ['GET','POST'])
def returvid():
    return render_template('index2.html')

@app.route('/output/<path:filename>')
def display(filename):
    folder_path = os.path.join(BASE_DIR, 'uploads')
    return send_from_directory(folder_path, filename)



if __name__ == '__main__':
    app.run()
