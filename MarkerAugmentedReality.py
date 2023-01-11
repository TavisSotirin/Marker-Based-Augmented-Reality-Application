import tkinter as tk
import tkinter.filedialog as fd
import sys
from enum import Enum

import cv2 as cv
import numpy as np
import numpy.linalg as ln
import math

class OverlayType(Enum):
    IMAGE = 0
    VIDEO = 1

def qPopup(title,message):
    if not tk.messagebox.askokcancel(title=title, message=message):
        print("User canceled setup.")
        sys.exit()
        
def selectFile(title):
    path = fd.askopenfilename(title=title)
    if path is None or path == "":
        print("No file selected.")
        sys.exit()
    return path

# Build the dot index array for the provided marker image to compare to any frame dot index calculated later on
def buildMarkerIndex(marker, blockSize=8, threshold=127):
    h,w = marker.shape
    markerDotIndex = []
    
    # Per 90 degree rotation
    for i in range(4):
        if i == 0:
            corners = np.array([[0,0],[0,w],[h,w],[h,0]])
        elif i == 1:
            corners = np.array([[0,w], [h,w], [h,0], [0,0]])
        elif i == 2:
            corners = np.array([[h,w],[h,0], [0,0], [0,w]])
        else:
            corners = np.array([[h,0],[0,0], [0,w], [h,w]])
        
        markerDotIndex.append(buildDotIndex(marker,corners,blockSize,threshold))
        
    return markerDotIndex

# Build dot index for a given image based on corners provided for the location of the contour being examined. A given contour is divided up into blockSize chunks for comparison.
def buildDotIndex(inImg, corners, blockSize=8, threshold=127):
    dotIndex = []
    midstep = 1.0/(blockSize*2)
    
    for row in range(blockSize):
        for col in range(blockSize):
            # Interpolation between corners to calculate % across current contour
            rowInterp = float(row)/blockSize + midstep
            colInterp = float(col)/blockSize + midstep
            
            # Calculate corners for a given block at row/col
            upper_x = corners[0][0] * (1 - rowInterp) + corners[1][0] * rowInterp
            lower_x = corners[3][0] * (1 - rowInterp) + corners[2][0] * rowInterp
            upper_y = corners[0][1] * (1 - rowInterp) + corners[1][1] * rowInterp
            lower_y = corners[3][1] * (1 - rowInterp) + corners[2][1] * rowInterp
            
            # Calculate pixel coords for the given block and check against mid value threshold. Store binary result
            dotIndex.append(inImg[round(upper_y * (1 - colInterp) + lower_y * colInterp)][round(upper_x * (1 - colInterp) + lower_x * colInterp)] >= threshold)
        
    return np.array(dotIndex)

# Compare a given dot index with the marker generated index, return true if they are similiar to a thresholded degree
def compareDotIndex(dotIndex1, dotIndex2, thresh=60):        
    if len(dotIndex1) == len(dotIndex2):
        return (dotIndex1 == dotIndex2).sum() >= thresh
    else:
        return False

def detectARCard(image, marker, markerIndex, minEdge=100):
    imgBW = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h,w = marker.shape

	# Find contours from threshold image to account/adapt for lighting in current frame.
    contours,_ = cv.findContours(cv.adaptiveThreshold(imgBW,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,10), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        # Use Douglas Peucker Algorithm to smooth each contour, without this each contour would likely have more than 4 points, making it difficult to identify which one may represent the marker in the frame
        closed = True
        ep = cv.arcLength(cnt,closed) * 0.01
        cntSmooth = cv.approxPolyDP(cnt, ep, closed)
        
        # Since the marker is square, only look at rectangular contours
        if cntSmooth.shape[0] == 4:
			# minEdge used to prevent super small contour line lengths
            if (cntSmooth[0][0][0] - cntSmooth[1][0][0])**2 + (cntSmooth[0][0][1] - cntSmooth[1][0][1])**2 > minEdge:
                contourDotIndex = buildDotIndex(imgBW, np.squeeze(cntSmooth, axis=1))
                found = False
                
                # Check each rotation of the marker image against the current contour, if one is a match, note it 
                if compareDotIndex(markerIndex[0], contourDotIndex):
                    found = True
                    corners = np.array([[0,0],[0,w], [h,w], [h,0]])
                elif compareDotIndex(markerIndex[1], contourDotIndex):
                    found = True
                    corners = np.array([[0,w], [h,w], [h,0], [0,0]])
                elif compareDotIndex(markerIndex[2], contourDotIndex):
                    found = True
                    corners = np.array([[h,w],[h,0], [0,0], [0,w]])
                elif compareDotIndex(markerIndex[3], contourDotIndex):
                    found = True
                    corners = np.array([[h,0],[0,0], [0,w], [h,w]])
                    
                # Return found corners and contour
                if found:
                    return corners.reshape(-1,1,2), cntSmooth.reshape(-1,1,2)

    return None, None
    
# Calculate R3 for the intrinsic matrix
def calcIntrinsic(intrinsic, estimated):
    dot = ln.inv(np.float64(intrinsic)).dot(np.float64(estimated))
    
    r1 = np.float64(dot[:, 0])
    r2 = np.float64(dot[:, 1])
    t = dot[:, 2]
    
	# Normalize using square of the product of the normals
    norm = 1.0 / np.float64(math.sqrt(np.float64(ln.norm(r1)) * np.float64(ln.norm(r2))))

    return np.array([r1,r2,np.cross(r1,r2)*norm,t]).transpose()

# Warp and set overlay image to overlap marker card if found
def applyOverlay(img, overlay, projection, template, origH):
    ogH,ogW = img.shape[:2]
    ovH,ovW = overlay.shape[:2]
    mH,mW = template.shape
    
    img = np.ascontiguousarray(img, dtype=np.uint8)
    
	# Mask out marker card
    mask = np.array([[0,0,0],[mW,0,0],[mW,mH,0],[0,mH,0]], np.float64).reshape(-1, 1, 3)
    mask = np.int32(cv.perspectiveTransform(mask, projection))
    
    cv.fillConvexPoly(img, mask, (0,0,0))
    
    # Resize overlay image to fit comfortably on marker card
    overlayScale = min(float(mH) / max(ovH,ovW), float(mW) / max(ovH,ovW))
    overlay = cv.resize(np.ascontiguousarray(overlay, dtype=np.uint8), (int(ovW*overlayScale),int(ovH*overlayScale)), interpolation=cv.INTER_CUBIC)
    
    # Calculate offset for non square overlay images
    repImgH, repImgW = overlay.shape[:2]
    minRep = min(repImgH, repImgW)
    offset = round((mH - minRep) / 2)
    
    # Add border to append offset to non square overlay images
    if minRep == repImgH:
        overlay = cv.copyMakeBorder(overlay, offset, 0, 0, 0, cv.BORDER_CONSTANT)
    else:
        overlay = cv.copyMakeBorder(overlay, 0, 0, offset, 0, cv.BORDER_CONSTANT)

    # Warp align overlay image to marker in frame
    warped = cv.warpPerspective(overlay, origH, (ogW,ogH))
    
    return cv.add(warped, img)

# Core body - access camera and loop every frame
def begin(marker_path, overlay_path, mSize=(480,480)):
    # Semi-generic camera properties derived from my camera (resolution of 1280x640)
    camera_parameters = np.array([[1000, 0, 640], [0, 1000, 320], [0, 0, 1]])

    # Open marker image
    marker = cv.imread(marker_path)
    overlay = cv.imread(overlay_path)
    overlayType = OverlayType.IMAGE
    
    if marker is None:
        assert False, "Invalid marker file selected."
    if overlay is None:
        assert False, "Invalid overlay file selected."

    # Resize marker and set to grayscale for matching
    marker = cv.resize(cv.flip(marker,1), mSize, interpolation=cv.INTER_CUBIC)
    marker = cv.cvtColor(marker,cv.COLOR_BGR2GRAY)

    # Open window and access camera
    cv.namedWindow("AR Cam",cv.WINDOW_AUTOSIZE)
    cv.startWindowThread()
    camera = cv.VideoCapture(0)
    
    if not camera.isOpened():
        assert False, "Failed to open camera."
        
    # Read initial camera frame
    hasFrame,_ = camera.read()

    # Build the dot index array for the provided marker image to compare to any frame dot index calculated later on
    markerIndex = buildMarkerIndex(marker)
    
    # Loop while camera is active or until escape is pressed
    while hasFrame:
        # Wait 20ms for key press, if escape(27), end program
        if cv.waitKey(20) == 27:
            camera.release()
            cv.destroyAllWindows()
            break
        
        # Get next camera frame
        hasFrame,frame = camera.read()
        
        # Locate the provided marker card image in the current frame, if possible
        ARCorners, frameCorners = detectARCard(frame, marker, markerIndex)
        
        # Calculate the homography matrix if the marker card was detected, otherwise just display the current frame
        if ARCorners is not None:
            HMat,_ = cv.findHomography(ARCorners, frameCorners, cv.RANSAC, 5.0)
        else:
            cv.imshow("AR Cam",np.flip(frame,axis=1))
            continue

        # Calculate intrinsic matrix/transformation matrix based on homography matrix and camera properties
        inMat = calcIntrinsic(camera_parameters, HMat)
        transform = camera_parameters.dot(inMat)
        
        # Process overlay image
        if overlayType == OverlayType.IMAGE:
            frame = np.flip(applyOverlay(frame, overlay, transform, marker, HMat), axis=1)
        elif overlayType == OverlayType.VIDEO:
            frame = np.flip(frame,axis=1)
            
        cv.imshow("AR Cam",frame)

if __name__ == '__main__':
    # Setup prompts for marker image and replacement image or video
    root = tk.Tk()
    root.withdraw()
    
    qPopup("Setup 1/2","Please choose the marker image to look for.")
    marker_path = selectFile("Select marker image")
    
    qPopup("Setup 2/2","Please choose the overlay image or video to replace the marker with.")
    overlay_path = selectFile("Select overlay file")
    
    root.destroy()
    
    begin(marker_path,overlay_path)