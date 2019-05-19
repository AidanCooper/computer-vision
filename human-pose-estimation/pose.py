import sys
import os
from argparse import ArgumentParser

import cv2
import time
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
        help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input",
        help="Path to video file or image. 'cam' for capturing video stream from internal camera.",
        required=True, type=str)
    parser.add_argument("-l", "--cpu_extension",
        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.",
        type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir",
        help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. "
             "Demo will look for a suitable plugin for device specified (CPU by default)",
        default="CPU", type=str)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    del net
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    cur_request_id = 0
    next_request_id = 1

    render_time = 0
    ret, frame = cap.read()
    while cap.isOpened():
        ret, next_frame = cap.read()
        if not ret:
            break
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        inf_start = time.time()
        in_frame = cv2.resize(next_frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs
            pw_relations = res['Mconv7_stage2_L1']
            kp_heatmaps = res['Mconv7_stage2_L2']
            """
            Nose 0, Neck 1, Right Shoulder 2, Right Elbow 3, Right Wrist 4,
            Left Shoulder 5, Left Elbow 6, Left Wrist 7, Right Hip 8,
            Right Knee 9, Right Ankle 10, Left Hip 11, Left Knee 12,
            LAnkle 13, Right Eye 14, Left Eye 15, Right Ear 16,
            Left Ear 17, Background 18
            """
            nPoints = 18
            POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],
                          [1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
                          [0,14],[0,15],[14,16],[15,17]]
            threshold = 0.2

            points = []
            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = kp_heatmaps[0, i, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
                # Scale the point to fit on the original image
                x = frame.shape[1] / probMap.shape[1] * point[0]
                y = frame.shape[0] / probMap.shape[0] * point[1]

                if prob > threshold:
                    if True: # Toggle circles and labels
                        cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                        cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                        cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(x), int(y)))
                else:
                    points.append(None)

            # Draw Skeleton
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)

            # Draw performance stats
            inf_time_message = "Inference time: N\A for async mode"
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
            async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id)

            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)

            # Resize output frame
            #frame = cv2.resize(frame, (1920, 1080))

        render_start = time.time()
        cv2.imshow("Detection Results", frame)
        render_end = time.time()
        render_time = render_end - render_start

        cur_request_id, next_request_id = next_request_id, cur_request_id
        frame = next_frame

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)
