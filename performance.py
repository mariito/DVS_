from __future__ import division
import math
import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
import matplotlib.pyplot as plt
import numpy as np
import getopt
import sys
import os
from glob import glob

#part of tis code has been obtained in
#https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py

def Resloss(img, imgname):   
	res_loss = os.path.getsize(imgname)/(img.shape[0]*img.shape[1])
	return res_loss

def readImage(filename):
	img = cv2.imread(filename, 0)
	if img is None:
		print('Invalid image:' + filename)
		return None
	else:
		return img

def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):
	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]

	coord_1 = []
	coord_2 = []
	
	cter = 0

	for mat in matches:

		# Get the matching keypoints for each of the images
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		# x - columns, y - rows
		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt
		if cter == 0:
			coord_1 = (x1,y1)
			coord_2 = (x2,y2)			
		else:
			coord_1 = np.vstack((coord_1, (int(x1),int(y1))))
			coord_2 = np.vstack((coord_2, (int(x2),int(y2))))

		cter += 1
	distx = abs(coord_1[0] - coord_2[0])
	disty = abs(coord_1[-1] - coord_2[-1])
	return distx,disty

# Runs sift algorithm to find features
def findFeatures(img):
	sift = cv2.xfeatures2d.SIFT_create()
	keypoints, descriptors = sift.detectAndCompute(img, None)
	
	return keypoints, descriptors

# Matches features given a list of keypoints, descriptors, and images
def matchFeatures(kp1, kp2, desc1, desc2, img1, img2):
	matcher = cv2.BFMatcher(cv2.NORM_L2, True)
	matches = matcher.match(desc1, desc2)
	dist_x, dist_y = drawMatches(img1,kp1,img2,kp2,matches)
	return dist_x, dist_y

def main():
	path_gt = ""#gt frames

	path_in = ""stabilized videos

	filesgt_ = sorted(glob(os.path.join(path_gt, '*')))
	filesin_ = sorted(glob(os.path.join(path_in, '*')))

	assert len(filesgt_) == len(filesin_)
	
	for vid in range(len(filesin_)): 
		counter = 0
		print(filesgt_[vid],filesin_[vid])
		png_files_gt = sorted(glob(os.path.join(filesgt_[vid], '*.jpg')))
		png_files_in = sorted(glob(os.path.join(filesin_[vid], '*.jpg')))
		txt_files_gt = sorted(glob(os.path.join(filesgt_[vid], '*.txt')))
		txt_files_in = sorted(glob(os.path.join(filesin_[vid], '*.txt')))

		pr1 = np.loadtxt(txt_files_gt[0])
		pr2 = np.loadtxt(txt_files_in[0])
		start_gt, extension = os.path.splitext(os.path.basename(png_files_gt[0])) 
		start_in, extension = os.path.splitext(os.path.basename(png_files_in[0])) 

		end_gt, extension = os.path.splitext(os.path.basename(png_files_gt[len(png_files_gt)-1]))
		end_in, extension = os.path.splitext(os.path.basename(png_files_in[len(png_files_in)-1]))
		print(vid,end_gt,end_in)
		assert start_in == start_gt, end_in == end_gt #comment this in case of discrepancies

		start = int(start_in)
		end = int(end_in)


		for ind in xrange(start, end + 1):

			img1name = png_files_gt[ind-1]
			img2name = png_files_in[ind-1]

			img1 = readImage(img1name)
			img2 = readImage(img2name)

			if img1 is not None and img2 is not None:
		
				#Resolution Loss 
				res_1 = Resloss(img1,img1name)
				res_2 = Resloss(img2,img2name)
				#MSE
				m_ = mse(img1, img2)
				#SSIM
				s_ = ssim(img1, img2)
				#MSE_PR
				mpr = mse(pr1[ind-1], pr2[ind-1])

				#performance metrics
				perf_ = [res_1/res_2, m_, s_, mpr]
				
				#dx,dy
				kp1, desc1 = findFeatures(img1)
				
				kp2, desc2 = findFeatures(img2)
				if (np.any(kp2==None)) or (np.any(desc2==None)):
					kp2 = kp1
					desc2 = desc1
				if (np.any(kp1==None)) or (np.any(desc1==None)):
					kp1 = kp2
					desc1 = desc2
				if ((np.any(kp2==None)) or (np.any(desc2==None)) and (np.any(kp1==None)) or (np.any(desc1==None))):
					kp1 = kp0
					kp2 = kp0
					desc1 = desc0
					desc2 = desc0
				else:
					kp0 = kp1
					desc0 = desc1
				keypoints = [kp1,kp2]

				dt_x, dt_y = matchFeatures(kp1, kp2, desc1, desc2, img1, img2)
				dt_ = [dt_x, dt_y]

				if counter == 0:
					perf = perf_
					dt = dt_
				else:
					perf = np.vstack((perf, perf_))
					dt = np.vstack((dt, dt_))

				counter += 1	
		dt_min_x = min(dt[:,0])
		dt_min_y = min(dt[:,-1])
		dt_max_x = max(dt[:,0])
		dt_max_y = max(dt[:,-1])

		for i in range(dt.shape[0]):
			dt2_x = (dt[i,0]-dt_min_x)/(dt_max_x-dt_min_x)
			dt2_y = (dt[i,-1]-dt_min_y)/(dt_max_y-dt_min_y)
			dt2_line = np.hstack((dt2_x,dt2_y))
			if i==0:
				dt2 = dt2_line
			else:
				dt2 = np.vstack((dt2, dt2_line))

		np.savetxt('Performance/GP/performance%03d.txt' %(vid), perf, fmt = '%.5f')
		np.savetxt('Performance/GP/dt%03d.txt' %(vid), dt, fmt = '%.5f')
		np.savetxt('Performance/GP/dt2%03d.txt' %(vid), dt2, fmt = '%.5f')

		perf_avg = np.mean(perf, axis=0)
		avg_dt = np.mean(dt, axis=0)
		avg_dt2 = np.mean(dt2, axis=0)
		if vid==0:
			perf_avg_all = perf_avg
			avg_dt_all = avg_dt
			avg_dt2_all = avg_dt2
		else:
			perf_avg_all = np.vstack((perf_avg_all,perf_avg))
			avg_dt_all = np.vstack((avg_dt_all,avg_dt))
			avg_dt2_all = np.vstack((avg_dt2_all,avg_dt2))

	np.savetxt('Performance/GP/performance_avg.txt', perf_avg_all, fmt = '%.5f')
	np.savetxt('Performance/GP/dt_avg.txt' , avg_dt_all, fmt = '%.5f')
	np.savetxt('Performance/GP/dt2_avg.txt' , avg_dt2_all, fmt = '%.5f')
if __name__ == "__main__":
	main()
