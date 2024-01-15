from ultralytics import YOLO
import cv2
import numpy as np

class overlay:
    def __init__(self, base, over):
        self.vbase = base
        self.vover = over
        self.model = YOLO('yolov8n-seg.pt')
    def run(self, boxes=False, t=128):
        while (True):
            ret1, bframe = self.vbase.read
            ret2, oframe = self.vover.read
            if ret1 and ret2:
                cv2.imshow("Test",
                           otest.composite(bframe, otest.edgedetect(oframe, otest.getBBOX(oframe), show_boxes=boxes), t))
            cv2.waitKey(1)

    #all of the following methods expect frame data to be in RGB color space (not grayscale)
    def getBBOX(self, frame): #returns a numpy array of every bounding box in xyxy
        #get frame results
        results = self.model(frame, stream=True)
        #build a list containing all the boxes
        allboxes = []
        for r in results:
            bcoords = r.boxes.xyxy
            for i in range(bcoords.shape[0]): #loop each row
                crow = [bcoords[i][0], bcoords[i][1], bcoords[i][2], bcoords[i][3]]
                allboxes.append(crow)
        #conversion to numpy array for faster processing later
        allboxes_np = np.empty((len(allboxes), 4))
        for i in range(len(allboxes)):
            allboxes_np[i][0] = allboxes[i][0]
            allboxes_np[i][1] = allboxes[i][1]
            allboxes_np[i][2] = allboxes[i][2]
            allboxes_np[i][3] = allboxes[i][3]
        return allboxes_np
    def edgedetect(self, frame, bboxes, show_boxes=False): #returns an image with an edge detection applied from the frame and chosen bounding boxes
        outlines = np.zeros_like(frame)
        for i in range(bboxes.shape[0]): #for every bounding box, update outline image
            #generate the "slice" of the image with our object, and edge-detects it
            x1 = int(bboxes[i][0])
            y1 = int(bboxes[i][1])
            x2 = int(bboxes[i][2])
            y2 = int(bboxes[i][3])
            if x1 >= x2 or y1 >= y2: continue #broken bboxes sometimes pop up
            #debug bbox show
            if show_boxes: cv2.rectangle(outlines, (x1, y1), (x2, y2), (255, 255, 255), 2)
            #edge detection
            fslice = frame[y1:y2, x1:x2]
            if fslice is None or fslice.size == 0: continue
            #automatically determine coefficients for canny edge detection depending on image parameters
            bfslice = cv2.blur(fslice, ksize=(5,5))
            med_val = np.nanmedian(bfslice)
            t1 = int(max(0, 0.7*med_val))
            t2 = int(min(255, 1.3*med_val))
            g_processed = cv2.Canny(fslice, t1, t2, L2gradient=True)
            if g_processed is None: continue
            #integrates into the main outline frame
            outlines_slice = outlines[y1:y2, x1:x2] #equivalent slice from current outlines
            g_outlines_slice = cv2.cvtColor(outlines_slice, cv2.COLOR_BGR2GRAY)
            g_comb_slice = np.maximum(g_outlines_slice, g_processed)
            outlines[y1:y2, x1:x2] = cv2.cvtColor(g_comb_slice, cv2.COLOR_GRAY2BGR)
        return outlines
    def composite(self, bframe, oframe, t=128): #composites the overlay (expected to be in grayscale but in RGB color space)
        #over the base frame, choosing high contrast colors as necessary
        #(threshold is for adjusting which intensity the overlay image has to be before it gets shown)
        for i in range(bframe.shape[0]):
            for j in range(bframe.shape[1]):
                b, g, r = oframe[i, j]
                av = (int(b)+int(g)+int(r))/3
                if av < t: continue #not enough for threshold, keeping original pixel data
                bcolor = av #"bright" overlay
                dcolor = 255-av #"dark" overlay
                bb, gg, rr = bframe[i, j]
                if (rr * 0.299 + gg * 0.587 + bb * 0.114) > 186: #the image is bright, use dark overlay
                    bframe[i, j] = (dcolor, dcolor, dcolor)
                else:
                    bframe[i, j] = (bcolor, bcolor, bcolor)
        return bframe

#TEST CODE
vtest = cv2.VideoCapture(0)
vtest.set(3, 640)
vtest.set(4, 480)
otest = overlay(vtest, vtest)
otest.run()