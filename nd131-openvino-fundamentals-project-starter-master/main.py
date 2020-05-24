"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import datetime

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    # Connect to the MQTT client
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def process(args):
    """
    Initialize the inference network, stream video to network, and output stats and video.
    :param args: Parsed Command line arguments
    :return: None
    """

    if args.input == '0':
        print('Camera stream not supported')
        return
    elif args.input.endswith('.png'):
        input_type = 'IMAGE'
    elif args.input.endswith('.mp4'):
        input_type = 'VIDEO'
    else:
        print('Image and video files are supported only')
        return
    # Only support two types of TF models. 
    if 'ssd' in args.model:
        model_type = 'SSD'
    elif 'faster_rcnn' in args.model:
        model_type = 'Faster-RCNN'
    else:
        print('Unknown model type')
        return

    # Initialise the class
    net = Network()

    # Load the model
    net.load_model(args.model)
    net_input_shape = net.get_input_shape()['image_tensor']
    net_shape = (net_input_shape[3], net_input_shape[2])

    #if input_type == 'IMAGE':
      #  process_image(args, net, net_shape, model_type)

    if input_type == 'VIDEO':
        process_stream(args, net, net_shape, model_type)

    cv2.destroyAllWindows()


def process_stream(args, infer_network, infer_network_shape, model_type):
    # Connect to the MQTT server
    client_mqtt = connect_mqtt()

    # Handle the input stream
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    frame_shape = (int(cap.get(3)), int(cap.get(4)))

    counter = 0
    duration = 0
    counter_prev = 0
    duration_prev = 0
    counter_total = 0
    counter_report = 0

    # Loop until stream is over
    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()
        if not flag:
            break

        # inference
        frame, cnt = infer_on_image(frame, frame_shape, model_type,
                                    infer_network, infer_network_shape,
                                    args.prob_threshold)

        # report duration only once (when person exits scene)
        duration_report = None

        # detection only if continues 3 frames or more
        if cnt != counter:
            counter_prev = counter
            counter = cnt
            if duration >= 3:
                duration_prev = duration
                duration = 0
            else:
                duration = duration_prev + duration
                duration_prev = 0  # unknown, not needed in this case
        else:
            duration += 1
            if duration >= 3:
                counter_report = counter
                if duration == 3 and counter > counter_prev:
                    # count as enter scene
                    counter_total += counter - counter_prev
                if duration == 3 and counter < counter_prev:
                    # count as exit scene, report duration in ms (note: FPS = 10)
                    duration_report = int((duration_prev / 10.0) * 1000)

        # Calculate and send relevant information on
        # current_count, total_count and duration to the MQTT server
        # Topic "person": keys of "count" and "total"
        # Topic "person/duration": key of "duration"
        client_mqtt.publish('person',
                            payload=json.dumps({
                                'count': counter_report, 'total': counter_total}),
                            qos=0, retain=False)
        if duration_report is not None:
            client_mqtt.publish('person/duration',
                                payload=json.dumps({'duration': duration_report}),
                                qos=0, retain=False)

        # Attention! Resize to cover for potential bug in the UI
        # Video size is 768x432, but the UI expects 758x432
        frame = cv2.resize(frame, (768, 432))

        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    cap.release()

"""
def process_image(args, infer_network, infer_network_shape, model_type):
    image = cv2.imread(args.input)
    image_shape = (image.shape[1], image.shape[0])
    image, cnt = infer_on_image(image, image_shape, model_type,
                                infer_network, infer_network_shape,
                                args.prob_threshold)
    cv2.imwrite('out.png', image)
"""

def infer_on_image(frame, frame_shape, model_type, infer_network, infer_network_shape, prob_threshold):
    # Pre-process the image
    net_image = cv2.resize(frame, infer_network_shape)
    net_image = net_image.transpose((2, 0, 1))
    net_image = net_image.reshape(1, *net_image.shape)

    # Start asynchronous inference
    if model_type == 'SSD':
        net_input = {
            'image_tensor': net_image
        }
    elif model_type == 'Faster-RCNN':
        net_input = {
            'image_tensor': net_image,
            'image_info': net_image.shape[1:]
        }

    num_detected = 0
    infer_network.exec_net(net_input)

    # Wait for the result
    if infer_network.wait() == 0:

        # Get the results of the inference request
        net_output = infer_network.get_output()

        # Extract any desired stats from the results
        # 1x1x100x7

        probs = net_output[0, 0, :, 2]
        for i, p in enumerate(probs):
            if p > prob_threshold:
                num_detected += 1
                box = net_output[0, 0, i, 3:]
                p1 = (int(box[0] * frame_shape[0]), int(box[1] * frame_shape[1]))
                p2 = (int(box[2] * frame_shape[0]), int(box[3] * frame_shape[1]))
                frame = cv2.rectangle(frame, p1, p2, (0, 0, 255), 3)

    # Return the number of detected objects (drawn boxes)
    return frame, num_detected


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    process(args)


if __name__ == '__main__':
    main()