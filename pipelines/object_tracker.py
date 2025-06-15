#!/usr/bin/env python

import time
import logging
import threading

import cv2

import depthai

import variables
from pipelines.pipeline import Pipeline


class ObjectTracker(Pipeline):
    def __init__(self):
        self.stop_event = threading.Event()

    def __session(self):
        # Create pipeline
        with depthai.Pipeline() as p:
            # Define sources and outputs
            camRgb = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_A)
            monoLeft = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_B)
            monoRight = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_C)

            stereo = p.create(depthai.node.StereoDepth)
            leftOutput = monoLeft.requestOutput((640, 400))
            rightOutput = monoRight.requestOutput((640, 400))
            leftOutput.link(stereo.left)
            rightOutput.link(stereo.right)

            spatialDetectionNetwork = p.create(depthai.node.SpatialDetectionNetwork).build(camRgb, stereo, "yolov6-nano")
            objectTracker = p.create(depthai.node.ObjectTracker)

            spatialDetectionNetwork.setConfidenceThreshold(0.7)
            spatialDetectionNetwork.input.setBlocking(False)
            spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
            spatialDetectionNetwork.setDepthLowerThreshold(100)
            spatialDetectionNetwork.setDepthUpperThreshold(5000)
            labelMap = spatialDetectionNetwork.getClasses()

            #objectTracker.setDetectionLabelsToTrack([0])  # track only person
            # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
            objectTracker.setTrackerType(depthai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
            # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
            objectTracker.setTrackerIdAssignmentPolicy(depthai.TrackerIdAssignmentPolicy.SMALLEST_ID)

            preview = objectTracker.passthroughTrackerFrame.createOutputQueue()
            tracklets = objectTracker.out.createOutputQueue()

            spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

            spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
            spatialDetectionNetwork.out.link(objectTracker.inputDetections)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            font_color = (0, 255, 0)
            p.start()
            logging.info("Object tracker initialised")
            while(p.isRunning()):
                while not self.stop_event.is_set():
                    imgFrame = preview.get()
                    track = tracklets.get()
                    assert isinstance(imgFrame, depthai.ImgFrame), "Expected ImgFrame"
                    assert isinstance(track, depthai.Tracklets), "Expected Tracklets"

                    counter+=1
                    current_time = time.monotonic()
                    if (current_time - startTime) > 1 :
                        fps = counter / (current_time - startTime)
                        counter = 0
                        startTime = current_time

                    frame = imgFrame.getCvFrame()
                    trackletsData = track.tracklets
                    for t in trackletsData:
                        roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                        x1 = int(roi.topLeft().x)
                        y1 = int(roi.topLeft().y)
                        x2 = int(roi.bottomRight().x)
                        y2 = int(roi.bottomRight().y)

                        try:
                            label = labelMap[t.label]
                        except:
                            label = t.label

                        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, font_color)
                        cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, font_color)
                        cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, font_color)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                        cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, font_color)
                        cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, font_color)
                        cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, font_color)

                    cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                    with variables.video_lock:
                        variables.video_frame = frame.copy()
                p.stop()
                logging.info("Object tracker stopped")


    def start(self):
        self.stop_event.clear()
        self.object_tracker_thread = threading.Thread(target=self.__session)
        self.object_tracker_thread.start()


    def stop(self):
        self.stop_event.set()
        self.object_tracker_thread.join()
        time.sleep(1)


    def exit(self):
        self.stop()