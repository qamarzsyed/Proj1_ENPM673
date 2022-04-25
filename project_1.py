import func
import numpy as np
import cv2

testudo = cv2.imread('testudo.png')
video = cv2.VideoCapture('1tagvideo.mp4')

stat, frame = video.read()

problem_1a = func.fft(frame)
cv2.imshow('afd', problem_1a)
cv2.waitKey(1000)

problem_2a = cv2.VideoWriter('Problem_2a.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))
problem_2b = cv2.VideoWriter('Problem_2b.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))

counter = 0
while stat:
    counter += 1
    if counter % 5 == 0:
        try:
            func.binary_code(func.binary_ar(frame))
            a_frame = func.apply_homography(testudo, func.homography(frame, testudo), frame)
            b_frame = func.draw_cube(frame)
            problem_2a.write(a_frame)
            problem_2b.write(b_frame)
        except:
            problem_2a.write(frame)
            problem_2b.write(frame)
    stat, frame = video.read()

problem_2a.release()
problem_2b.release()

