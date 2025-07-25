#!/bin/bash

nvcc Lab10.cu -o Lab10 -Xcompiler -Wno-overloaded-virtual -arch=sm_75 \
-I/usr/include/opencv4 \
-lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib \
-lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_face -lopencv_freetype \
-lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor \
-lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo \
-lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow \
-lopencv_surface_matching -lopencv_tracking -lopencv_highgui -lopencv_datasets -lopencv_text \
-lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_ximgproc \
-lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs \
-lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_xphoto -lopencv_photo \
-lopencv_imgproc -lopencv_core
