# Marker-Based-Augmented-Reality-Application
Simple marker based augmented reality script written in Python

The application converts a known marker image into a provided overlay image in real time using the systems camera.

The goal of this project was to create a simple augmented reality application that, given a predefined 
image (referred to as the ‘marker’), could identify said image in real time through the view of a camera 
(such as a webcam, with ‘real time’ referring to the frame rate of the camera), and replace it with a 
separate image (referred to as the ‘overlay’). The replacement image would match the orientation and 
perspective of the marker in the camera’s world space.

While the application performs about as intended, and accomplishes the initial goal that was set, there 
is room for dramatic improvement. One of the most promising improvements would be utilizing a smart 
algorithm for tracking movement, such as the Lucas–Kanade Sparse Optical Flow algorithm. This would 
help to remove the jitter between frames that have a strong marker reading and would also help with 
moving the marker around quicker while maintaining projection.
