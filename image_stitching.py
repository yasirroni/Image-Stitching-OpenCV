import cv2
import numpy as np
import sys

class ImageStitching():
    # This refactor / major update is done by the help of:
    # 1. https://github.com/sykrn
    def __init__(self, ratio=0.85, min_match=10, horizontal=False, feats='ORB', method='H'):
        """
        feats:
            ORB: using cv2.ORB_create()
            SIFT: using cv2.xfeatures2d.SIFT_create()
            ORB-SIFT: using cv2.ORB_create(), if H is None use cv2.xfeatures2d.SIFT_create()
        method:
            y: shift in y coordinate
            x: NOT YET
            xy: NOT YET
            H: map using matrix multiplication H @ IMG2
        """
        self.ratio = ratio
        self.min_match = min_match
        self.horizontal = horizontal
        self.method = method
        
        if feats == 'ORB': # faster, less accurate
            self.feats = cv2.ORB_create()
            self.feats2 = None
        elif feats == 'SIFT': # slower, more accurate
            self.feats = cv2.xfeatures2d.SIFT_create()
            self.feats2 = None
        elif feats == 'ORB-SIFT':
            self.feats = cv2.ORB_create()
            self.feats2 = cv2.xfeatures2d.SIFT_create()
        else:
            # TODO:
            # Support ORB-SIFT, if ORB fail use SIFT
            print(f"Not known feats method of {feats}")

    def detect(self, img1, img2, m1=None, m2=None):
        kp1, des1 = self.feats.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), m1)
        kp2, des2 = self.feats.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), m2)
        return kp1, des1, kp2, des2
    
    def detect2(self, img1, img2, m1=None, m2=None):
        kp1, des1 = self.feats2.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), m1)
        kp2, des2 = self.feats2.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), m2)
        return kp1, des1, kp2, des2
    
    def get_good_points(self, des1, des2):
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
      
        good_points = []
        # good_matches=[] # unused
        for m1, m2 in raw_matches: # what is m1 and m2 here?
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                # good_matches.append([m1]) # unused
        return good_points
        
    def get_H(self, kp1, kp2, good_points):
        image1_kp = np.float32(
            [kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32(
            [kp2[i].pt for (i, _) in good_points])

        H, _ = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0) # outpur: H, status
        return H
    
    def _get_H_wrapper(self, kp1, kp2, des1, des2, y_shift_bounds=(-np.inf, np.inf)):
        good_points = self.get_good_points(des1, des2)
        
        if len(good_points) > self.min_match:
            H = self.get_H(kp1, kp2, good_points)
            if H is not None and self.method == 'y':
                if y_shift_bounds[0] <= H[1,2] <= y_shift_bounds[1]:
                    y_shift = int(H[1,2])
                    H = np.identity(3)
                    H[1,2] = y_shift
                else:
                    H = None
        else:
            H = None
        return H
    
    def stitch(self, img1, img2, H=None, y_shift_bounds=(-np.inf, np.inf), m1=None, m2=None, blending=False):
        if H is None:
            kp1, des1, kp2, des2 = self.detect(img1, img2, m1, m2)
            H = self._get_H_wrapper(kp1, kp2, des1, des2, y_shift_bounds)
            if H is None and self.feats2:
                kp1, des1, kp2, des2 = self.detect2(img1, img2, m1, m2)
                H = self._get_H_wrapper(kp1, kp2, des1, des2, y_shift_bounds)
        
        if H is None:
            return img1, None

        if self.method == 'H':
            height_img1 = img1.shape[0]
            width_img1 = img1.shape[1]
            height_panorama = height_img1 + (img2.shape[0] if not self.horizontal else 0)
            width_panorama = width_img1  + (img2.shape[1] if self.horizontal else 0)
            
            # TODO:
            # support blending
            # if blending:
            #     panorama1 = np.zeros((height_panorama, width_panorama, 3))
            #     blending_mask1 = self.create_blending_mask(img1,img2,version='left_image')
            #     panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
            #     panorama1 *= blending_mask1
            #     blending_mask2 = self.create_blending_mask(img1,img2,version='right_image')

            #     panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
            #     panorama = panorama1+panorama2

            panorama = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))       
            panorama[:height_img1,:width_img1] = img1

            rows, cols = np.where(panorama[:, :, 0] != 0)

            min_row, max_row = rows.min(), rows.max() + 1
            min_col, max_col = cols.min(), cols.max() + 1
            img1 = panorama[min_row:max_row, min_col:max_col, :]
        elif self.method == 'y':
            # TODO: support self.horizontal
            height_img1 = img1.shape[0]
            
            # TODO:
            # support blending
            img1 = np.vstack((img1, img2[int(height_img1-H[1,2]):,:,:]))
            
            # TODO: support preserve second image instead of first image 
            # img1 = np.vstack((img1[:int(height_img1-H[1,2]),:,:], img2))
        else:
            print('UNKNOWN METHOD')
        return img1, H
    
    def stitch_from_paths(self, img_path_list, H=None, y_shift_bounds=(-np.inf, np.inf), m1=None, m2=None, blending=False):
        # TODO: support masks
        img1 = cv2.imread(img_path_list[-1])
        if H is None:
            H = []
            out_img_path_list = [img_path_list[-1]]
            for i in range(len(img_path_list)-2,-1,-1):
                img2 = cv2.imread(img_path_list[i])
                img1, H_ = self.stitch(img1, img2, m1=m1, m2=m2, blending=blending)
                if type(H_) == type(None):
                    break
                else:
                    H.append(H_)
                    out_img_path_list.append(img_path_list[i])
            return img1, H, list(out_img_path_list)
        else:
            for i, H_ in zip(range(len(img_path_list)-2,-1,-1), H):
                img2 = cv2.imread(img_path_list[i])
                img1, _ = self.stitch(img1, img2, H_, y_shift_bounds, m1=m1, m2=m2, blending=blending)
            return img1, H, list(reversed(img_path_list))

            
# TODO: fully support mask method
def create_mask(img, bboxes):    
    y,x = img.shape[:2]    
    mask = np.zeros((y,x),dtype='uint8')
    
    for bbox in bboxes:
        xmin,ymin,xmax,ymax = list(bbox)
        mask[ymin:ymax,:] = 1 # all columns
    
    return mask

# TODO: support blending
# def create_blending_mask(self,img1,img2,version):
#     height_img1 = img1.shape[0]
#     width_img1 = img1.shape[1]
#     width_img2 = img2.shape[1]
#     height_panorama = height_img1
#     width_panorama = width_img1 +width_img2
#     offset = int(self.smoothing_window_size / 2)
#     barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
#     mask = np.zeros((height_panorama, width_panorama))
#     if version== 'left_image':
#         mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
#         mask[:, :barrier - offset] = 1
#     else:
#         mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
#         mask[:, barrier + offset:] = 1
#     return cv2.merge([mask, mask, mask])

def main(argv1, argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    final=ImageStitching().stitch(img1,img2)
    cv2.imwrite('panorama.jpg', final)

if __name__ == '__main__':
    try: 
        main(sys.argv[1], sys.argv[2])
    except IndexError:
        print ("Please input two source images: ")
        print ("For example: python image_stitching.py '/Users/linrl3/Desktop/picture/p1.jpg' '/Users/linrl3/Desktop/picture/p2.jpg'")
