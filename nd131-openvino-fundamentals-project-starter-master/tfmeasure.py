import datetime

import tensorflow as tf
import numpy as np
import cv2


def load_model(graph_file):
    with tf.io.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def init_session(graph_def):
    # Create session and load graph
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.compat.v1.Session(config=tf_config)
    tf.import_graph_def(graph_def, name='')
    return tf_sess


def infer(tf_sess, image):
    # get dimensions
    input_names = ['image_tensor']
    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

    image_net = cv2.resize(image, (300, 300))

    scores, boxes, classes, num_detections = tf_sess.run(
        [tf_scores, tf_boxes, tf_classes, tf_num_detections],
        feed_dict={tf_input: image_net[None, ...]}
    )

    boxes = boxes[0]  # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])
    print('************************************ ', num_detections)

    box = boxes[0]

    width, height, _ = image.shape

    p1 = (int(box[1] * height), int(box[0] * width))
    p2 = (int(box[3] * height), int(box[2] * width))
    return cv2.rectangle(image, p1, p2, (0, 0, 255), 3)


def convert_image(tf_sess, file_in, file_out):
    image = cv2.imread(file_in)
    image_out = infer(tf_sess, image)
    cv2.imwrite(file_out, image_out)
    cv2.destroyAllWindows()


def convert_video(tf_sess, file_in, file_out):
    cap = cv2.VideoCapture(file_in)
    cap.open(file_in)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Video out
    out = cv2.VideoWriter(file_out, 0x00000021, 30, (width, height))

    t_start = datetime.datetime.now()
    t_infer = []
    print('### Start:', t_start)

    # Process frames until the video ends, or process is exited
    frame_count = 0
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break

        out.write(image)
        frame_count += 1
        print('.', end='', flush=True)
        if frame_count % 100 == 0:
            print()

    t_end = datetime.datetime.now()
    print()
    print('### Finished:', t_end)
    print('### Video processed in:', (t_end - t_start))

  

    cv2.destroyAllWindows()
    cap.release()
    out.release()


# graph_file_ssd = 'model/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
graph_file_faster_rcnn = 'model/frozen_inference_graph.pb'

input_file_video = 'resources/Pedestrian_Detect_2_1_1.mp4'
# input_file_image = 'data/people-counter-image.png'


def main():
    model = load_model(graph_file_faster_rcnn)
    session = init_session(model)
    # convert_image(session, input_file_image, "resources/image-out.png")
    convert_video(session, input_file_video, "resources/video-out-faster-rcnn.mp4")


if __name__ == '__main__':
    main()