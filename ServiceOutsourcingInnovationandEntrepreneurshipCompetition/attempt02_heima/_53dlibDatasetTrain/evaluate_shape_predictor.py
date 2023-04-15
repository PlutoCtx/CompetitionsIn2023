# @Version: python3.10
# @Time: 2023/4/15 16:13
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: evaluate_shape_predictor.py
# @Software: PyCharm
# @User: chent

# USAGE
# python evaluate shape predictor.py --predictor eye_predictor.dat --xml ibug_300W_large_face_landmark_data
# python evaluate shape predictor py --predictor eye_predictor.dat --xml ibug_300W_large_face_landmark_data
# import the necessary packages
import argparse
import dlib

# construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predictor", required=True,
                help="path to trained dlib shape predictor model")
ap.add_argument("-x", "--xml", required=True,
                help="path to input training/testing XML file")
args = vars(ap.parse_args())

# compute the error over the supplied data split and display it to
# our screen
print("[INFO] evaluating shape predictor...")
error = dlib.test_shape_predictor(args["xml"], args["predictor"])
print("[INFO] error: {}".format(error))