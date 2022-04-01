import cv2 as cv
import pickle
import timeit

start = timeit.default_timer()

im = cv.imread("grid.jpg")
index = pickle.loads(open("keypoints.txt", "rb").read())
kp2 = []
des2 = pickle.loads(open("descriptors.txt", "rb").read())
for point in index:
    temp = cv.KeyPoint(x=point[0][0],y=point[0][1],size=point[1], angle=point[2], response=point[3], octave=point[4], class_id=point[5])
    kp2.append(temp)
imm = cv.drawKeypoints(im, kp2, outImage=None)




img1 = cv.imread('rotated.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('grid.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)


# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
correct = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        correct.append(m)
points = []
for match in correct:
    p1 = kp1[match.queryIdx].pt
    p2 = kp2[match.trainIdx].pt
    points.append(p2)
gridLocation = cv.imread("gridPlacement.jpeg", cv.IMREAD_UNCHANGED)
grid_space = []
for point in points:
    x = int(point[0]/400)+1
    y = int(point[1]/400)
    grid_space.append(chr(y+65)+str(x))
    cv.circle(gridLocation, center=(int(point[0]),int(point[1])), radius=10, color=(0, 0, 255), thickness=-1)
location_output = []
[location_output.append(x) for x in grid_space if x not in location_output]
print(location_output)
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
img3 = cv.resize(img3, (500, 500), interpolation=cv.INTER_AREA)

stop = timeit.default_timer()
gridLocation = cv.resize(gridLocation, (500, 500), interpolation=cv.INTER_AREA)

print('Time: ', stop - start)
cv.imshow('placement',gridLocation)
cv.imshow('hi',img3)
cv.waitKey(0)

