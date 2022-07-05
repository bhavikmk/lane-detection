import imp
import cv2 
import numpy as np

video = cv2.VideoCapture('/home/bhavik/projects/compVision/lane_detection/Highway - 10364.mp4')

while(video.isOpened()):
    ret, img = video.read()
    if ret == True:
        
        # Fill lane detection algotithm


        # kernel = np.ones((5,5),np.float32)/25
        # dst = cv2.filter2D(img,-1,kernel)

        blur = cv2.GaussianBlur(img,(5,5),0)
        edges = cv2.Canny(blur,100,200)

        mask = np.zeros_like(edges)   
        ignore_mask_color = 255

        imshape = img.shape
        vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        rho = 1
        theta = np.pi/180
        threshold = 1
        min_line_length = 10
        max_line_gap = 1

        line_image = np.copy(masked_edges)*0 #creating a blank to draw lines on

        # Run Hough on edge detected image
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)

        # Iterate over the output "lines" and draw lines on the blank
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(masked_edges,(x1,y1),(x2,y2),(255,0,0),5)

        # Draw the lines on the edge image
        combo = cv2.addWeighted(masked_edges, 0.8, masked_edges, 1, 0) 
        cv2.imshow("Frame", combo)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        
video.release()
cv2.destroyAllWindows()    