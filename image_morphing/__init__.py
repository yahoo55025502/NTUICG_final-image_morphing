__version__ = '0.1.0'
import cv2
import dlib
import numpy as np
import argparse
import sys
import time
from subprocess import Popen, PIPE
from PIL import Image

# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def generate_face_correspondences(img1, img2):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("image_morphing/shape_predictor_68_face_landmarks_GTX.dat")
    corresp = np.zeros((68,2))

    imgList = crop_image(img1, img2)
    img_features = []

    for idx, img in enumerate(imgList):
        temp_list = []
        size = (img.shape[0],img.shape[1])
        dets = detector(img, 1)
        if len(dets) == 0:
            if idx == 0:
                sys.exit("face detection in first image is failed.")
            if idx == 1:
                sys.exit("face detection in second image is failed.")
        for k, rect in enumerate(dets):
            shape = predictor(img, rect)
            
            for i in range(0,68):
                x = shape.part(i).x
                y = shape.part(i).y
                corresp[i][0] += x
                corresp[i][1] += y
                temp_list.append((x, y))
                
            temp_list.append((1,1))
            temp_list.append((size[1]-1,1))
            temp_list.append(((size[1]-1)//2,1))
            temp_list.append((1,size[0]-1))
            temp_list.append((1,(size[0]-1)//2))
            temp_list.append(((size[1]-1)//2,size[0]-1))
            temp_list.append((size[1]-1,size[0]-1))
            temp_list.append(((size[1]-1),(size[0]-1)//2))
        img_features.append(temp_list)
    mid_features = corresp/2
    mid_features = np.append(mid_features,[[1,1]],axis=0)
    mid_features = np.append(mid_features,[[size[1]-1,1]],axis=0)
    mid_features = np.append(mid_features,[[(size[1]-1)//2,1]],axis=0)
    mid_features = np.append(mid_features,[[1,size[0]-1]],axis=0)
    mid_features = np.append(mid_features,[[1,(size[0]-1)//2]],axis=0)
    mid_features = np.append(mid_features,[[(size[1]-1)//2,size[0]-1]],axis=0)
    mid_features = np.append(mid_features,[[size[1]-1,size[0]-1]],axis=0)
    mid_features = np.append(mid_features,[[(size[1]-1),(size[0]-1)//2]],axis=0)
    return (img_features[0],img_features[1],mid_features)

def get_triangle_list(rect_w, rect_h, feature_pts):
    rect = (0, 0, rect_w, rect_h)
    subdiv = cv2.Subdiv2D(rect)
    feature_pts_list = feature_pts.tolist()
    feature_pts = [(int(x[0]), int(x[1])) for x in feature_pts_list]
    dict = {x[0]:x[1] for x in list(zip(feature_pts, range(76)))}
    for pt in feature_pts:
        subdiv.insert(pt)
    triangleList = subdiv.getTriangleList()
    triangleList_in_idx = []
    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3) :
            triangleList_in_idx.append((dict[pt1],dict[pt2],dict[pt3]))
    return triangleList_in_idx

def triangle_morphing(img1, img2, img, img1_tri, img2_tri, mid_tri, alpha):
    
    # get bounding rectangles of three triangles
    r1 = cv2.boundingRect(np.float32([img1_tri]))
    r2 = cv2.boundingRect(np.float32([img2_tri]))
    r = cv2.boundingRect(np.float32([mid_tri]))

    # create mask that will be occupied by (1.0, 1.0, 1.0) in triangle region, and occupied by (0.0, 0.0, 0.0) in non-triangle region.
    t1Rect = []
    t2Rect = []
    tRect = []
    for i in range(0, 3):
        t1Rect.append(((img1_tri[i][0] - r1[0]),(img1_tri[i][1] - r1[1])))
        t2Rect.append(((img2_tri[i][0] - r2[0]),(img2_tri[i][1] - r2[1])))
        tRect.append(((mid_tri[i][0] - r[0]),(mid_tri[i][1] - r[1])))
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)
    
    # get image 1 and image 2 regions in bounding rectangles
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpMat = cv2.getAffineTransform(np.float32(t1Rect), np.float32(tRect))
    warpImage1 = cv2.warpAffine(img1Rect, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warpMat = cv2.getAffineTransform(np.float32(t2Rect), np.float32(tRect))
    warpImage2 = cv2.warpAffine(img2Rect, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
    
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def morphing(img1, img2, feature_pts_1, feature_pts_2, midTriangles, output):
    duration = 3
    frame_rate = 10
    num_imgs = int(duration*frame_rate)
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate),'-s',str(img1.shape[1])+'x'+str(img1.shape[0]), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', output], stdin=PIPE)
    for f in range(num_imgs):
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        pts = []
        alpha = f / (num_imgs-1)

        for i in range(0, len(feature_pts_1)):
            x = (1 - alpha) * feature_pts_1[i][0] + alpha * feature_pts_2[i][0]
            y = (1 - alpha) * feature_pts_1[i][1] + alpha * feature_pts_2[i][1]
            pts.append((x,y))

        frame_morphed = np.zeros(img1.shape, dtype = img1.dtype)

        for i in range(len(midTriangles)):
            idx0 = int(midTriangles[i][0])
            idx1 = int(midTriangles[i][1])
            idx2 = int(midTriangles[i][2])

            img1_tri = [feature_pts_1[idx0], feature_pts_1[idx1], feature_pts_1[idx2]]
            mid_tri = [pts[idx0], pts[idx1], pts[idx2]]
            img2_tri =  [feature_pts_2[idx0], feature_pts_2[idx1], feature_pts_2[idx2]]

            triangle_morphing(img1, img2, frame_morphed, img1_tri, img2_tri, mid_tri, alpha)

        res = Image.fromarray(cv2.cvtColor(np.uint8(frame_morphed), cv2.COLOR_BGR2RGB))
        res.save(p.stdin,'JPEG')     
    p.stdin.close()
    p.wait()  

# the following three function are copied from https://github.com/Azmarie/Face-Morphing, only using for rescale images
def calculate_margin_help(img1,img2):
    size1 = img1.shape
    size2 = img2.shape
    diff0 = abs(size1[0]-size2[0])//2
    diff1 = abs(size1[1]-size2[1])//2
    avg0 = (size1[0]+size2[0])//2
    avg1 = (size1[1]+size2[1])//2

    return [size1,size2,diff0,diff1,avg0,avg1]

def crop_image(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)

    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        scale0 = size1[0]/size2[0]
        scale1 = size1[1]/size2[1]
        if(scale0 > scale1):
            res = cv2.resize(img2,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img2,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(img1,res)

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        scale0 = size2[0]/size1[0]
        scale1 = size2[1]/size1[1]
        if(scale0 > scale1):
            res = cv2.resize(img1,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img1,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(res,img2)

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]
    
    else:
        return [img1[:,diff1:avg1],img2[-diff0:avg0,:]]

def crop_image_help(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)
    
    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        return [img1,img2[-diff0:avg0,-diff1:avg1]]

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        return [img1[diff0:avg0,diff1:avg1],img2]

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]

    else:
        return [img1[:,diff1:avg1],img2[-diff0:avg0,:]]

def main():
    parser = argparse.ArgumentParser(description='image_morphing')
    parser.add_argument('-a', type=str, dest="path1", help="first input image path")
    parser.add_argument('-b', type=str, dest="path2", help="second input image path")
    parser.add_argument('-o', type=str, dest="output", help="output path")
    args = parser.parse_args()
    
    img1 = cv2.imread(args.path1)
    img2 = cv2.imread(args.path2)
    feature_pts_img1, feature_pts_img2, mid_features = generate_face_correspondences(img1, img2)
    size = img1.shape
    triangleList = get_triangle_list(size[1], size[0], mid_features)
    morphing(img1, img2, feature_pts_img1, feature_pts_img2, triangleList, args.output)

    
if __name__ == '__main__':
    main()


