#!/usr/bin/env python

import time
import logging
import threading

import cv2

import depthai

import variables
from pipelines.pipeline import Pipeline

class AprilTag(Pipeline):
    def __init__(self):
        self.stop_event = threading.Event()
        self.color = (0, 255, 0)
        self.start_time = time.monotonic()
        self.counter = 0
        self.fps = 0.0

    def __session(self):
        with depthai.Pipeline() as p:
            host_camera = p.create(depthai.node.Camera).build()
            apriltag_node = p.create(depthai.node.AprilTag)
            host_camera.requestOutput((1280, 720)).link(apriltag_node.inputImage)
            passthrough_output_queue = apriltag_node.passthroughInputImage.createOutputQueue()
            output_queue = apriltag_node.out.createOutputQueue()

            p.start()
            logging.info("AprilTag tracker initialised")
            while p.isRunning():
                while not self.stop_event.is_set():
                    apriltag_message = output_queue.get()
                    assert(isinstance(apriltag_message, depthai.AprilTags))
                    tags = apriltag_message.aprilTags

                    self.counter += 1
                    current_time = time.monotonic()
                    if (current_time - self.start_time) > 1:
                        self.fps = self.counter / (current_time - self.start_time)
                        counter = 0
                        self.start_time = current_time

                    passthrough_image: depthai.ImgFrame = passthrough_output_queue.get()
                    frame = passthrough_image.getCvFrame()

                    def to_int(tag):
                        return (int(tag.x), int(tag.y))

                    for tag in tags:
                        top_left = to_int(tag.topLeft)
                        top_right = to_int(tag.topRight)
                        bottom_right = to_int(tag.bottomRight)
                        bottom_left = to_int(tag.bottomLeft)

                        center = (int((top_left[0] + bottom_right[0]) / 2), int((top_left[1] + bottom_right[1]) / 2))

                        cv2.line(frame, top_left, top_right, self.color, 2, cv2.LINE_AA, 0)
                        cv2.line(frame, top_right,bottom_right, self.color, 2, cv2.LINE_AA, 0)
                        cv2.line(frame, bottom_right,bottom_left, self.color, 2, cv2.LINE_AA, 0)
                        cv2.line(frame, bottom_left,top_left, self.color, 2, cv2.LINE_AA, 0)

                        idStr = "ID: " + str(tag.id)
                        cv2.putText(frame, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.color)

                        cv2.putText(frame, f"fps: {self.fps:.1f}", (200, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.color)

                    #cv2.imshow("detections", frame)
                    with variables.video_lock:
                        variables.video_frame = frame.copy()

                p.stop()
                logging.info("AprilTag tracker stopped")


    def start(self):
        self.stop_event.clear()
        self.apriltag_thread = threading.Thread(target=self.__session)
        self.apriltag_thread.start()


    def stop(self):
        self.stop_event.set()
        self.apriltag_thread.join()
        time.sleep(1)


    def exit(self):
        self.stop()
