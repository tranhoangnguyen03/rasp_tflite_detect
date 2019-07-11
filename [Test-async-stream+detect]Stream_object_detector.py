import numpy as np
import tensorflow as tf
import cv2
import os
import time
from multiprocessing import Process
from queue import Queue
import threading

from object_detector_detection_api import ObjectDetectorDetectionAPI, \
                                          PATH_TO_LABELS, NUM_CLASSES

from videocaptureasync import VideoCaptureAsync


class ObjectDetectorLite(ObjectDetectorDetectionAPI):
    def __init__(self, model_path='detect.tflite'):
        """
            Builds Tensorflow graph, load model and labels
        """

        # Load lebel_map
        self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

        # Define lite graph and Load Tensorflow Lite model into memory
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, image, threshold=0.1):
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """

        # Resize and normalize image for network input
        frame = cv2.resize(image, (300, 300))
        frame = np.expand_dims(frame, axis=0)
        frame = (2.0 / 255.0) * frame - 1.0
        frame = frame.astype('float32')

        # run model
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()

        # get results
        boxes = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])
        num = self.interpreter.get_tensor(
            self.output_details[3]['index'])

        # Find detected boxes coordinates
        return self._boxes_coordinates(image,
                            np.squeeze(boxes[0]),
                            np.squeeze(classes[0]+1).astype(np.int32),
                            np.squeeze(scores[0]),
                            min_score_thresh=threshold)

    def close(self):
        pass

def process_frame(detector, inputQueue, outputQueue):
	# keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            frame = inputQueue.get()
            result = detector.detect(frame, 0.5)
			# write the detections to the output queue
            outputQueue.put(result)

if __name__ == '__main__':
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    font = cv2.FONT_HERSHEY_SIMPLEX

    detector = ObjectDetectorLite()

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
	
	# initialize the input queue (frames), output queue (detections),
    # and the list of actual detections returned by the child process
    inputQueue = Queue(maxsize=1)
    outputQueue = Queue(maxsize=1)
    result = None
    print('Queue started')
	
    source = "rtmp://192.168.0.115/live"
    cap = VideoCaptureAsync(src=source)
    cap.start()
    time.sleep(2)

    p = threading.Thread(target=process_frame, args=(detector, inputQueue,outputQueue,))
    #p.daemon = True
    p.start()
    
    while True:
        t1 = cv2.getTickCount()

        _, frame = cap.read()	

        if inputQueue.empty():  ## if queue is empty
            inputQueue.put(frame) ## pass frame onto inputQueue for proces_frame
        if not outputQueue.empty(): ## if outputQueue is empty
            result = outputQueue.get() ## retrieve frame from outputQueue
        if result:          
            for obj in result:
                if obj[3] == "person":
                    cv2.rectangle(frame, obj[0], obj[1], (0, 255, 0), 2)
                    cv2.putText(frame, '{}: {:.2f}'.format(obj[3], obj[2]),
                                (obj[0][0], obj[0][1] - 5),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        cv2.putText(frame,"Frame Count: {0:.2f}".format(cap.frame_count),
                    (30,100),font,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,"FPS: {0:.2f}".format(1/cap.time_per_frame),
                    (30,50),font,1,(255,255,0),2,cv2.LINE_AA)
            ### Showing frame
        cv2.imshow('Object detector', frame)
            ### FPS calculations
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        print('time_per_frame =', time1)
        if cv2.waitKey(1) == ord('q'):
             break
    cv2.destroyAllWindows()
    cap.stop()
    inputQueue.task_done()
    outputQueue.task_done()
    p.join()




"""	
    while(True):
        t1 = cv2.getTickCount()

        _, frame = cap.read()
        result = detector.detect(frame, 0.5)
        for obj in result:
            if obj[3] == "person":
                cv2.rectangle(frame, obj[0], obj[1], (0, 255, 0), 2)
                cv2.putText(frame, '{}: {:.2f}'.format(obj[3], obj[2]),
                            (obj[0][0], obj[0][1] - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        cv2.putText(frame,"Frame Count: {0:.2f}".format(cap.frame_count),
                    (30,100),font,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),
                    (30,50),font,1,(255,255,0),2,cv2.LINE_AA)
            ### Showing frame
        cv2.imshow('Object detector', frame)
            ### FPS calculations
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        print('time_per_frame =', time1)
        if cv2.waitKey(1) == ord('q'):
             break


cap.stop()
cv2.destroyAllWindows()
"""


"""    while(True):
        t1 = cv2.getTickCount()

        ret = camera.grab()

        if frame_count % 15 == 0: # skip every n frames
            ret, frame = camera.retrieve()
            if ret:
        ##################################################################
            ### Frame processing & detection
                Uframe = cv2.UMat(frame)
                frame_expanded = np.expand_dims(Uframe, axis=0)

                result = detector.detect(frame, 0.5)
                for obj in result:
                   ### print('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                   ###       format(obj[0], obj[1], obj[3], obj[2]))
            ### Writing onto frame
                    if obj[3] == "person":
                        cv2.rectangle(frame, obj[0], obj[1], (0, 255, 0), 2)
                        cv2.putText(frame, '{}: {:.2f}'.format(obj[3], obj[2]),
                                    (obj[0][0], obj[0][1] - 5),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),
                                (30,50),font,1,(255,255,0),2,cv2.LINE_AA)
            ### Showing frame
                cv2.imshow('Object detector', frame)
            ### FPS calculations
                t2 = cv2.getTickCount()
                time1 = (t2-t1)/freq
                frame_rate_calc = 1/time1
            else:
                print("No stream signal")
                camera = cv2.VideoCapture("rtmp://192.168.137.1/live", cv2.CAP_FFMPEG)
                time.sleep(1)

        frame_count +=1

        if cv2.waitKey(1) == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
    detector.close()"""
