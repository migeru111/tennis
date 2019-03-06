import cv2
import sys
import numpy as np
import math
import statistics
from matplotlib import pyplot as plt

usevideo="1.mp4"

'''
import skimage.data
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.util import invert
from skimage import io
import skimage.data
import skimage.color
'''

'''
##############################

pts コート上の各点を格納 np.arrayに格納
###############courtpoint####################
#####[0]コート左上
#####[1]コート右上
#####[2]コート右下
#####[3]コート左下
#####[4]上部左シングルコート点
#####[5]下部左シングルコート点
#####[6]上部右シングルコート点
#####[7]下部右シングルコート点
#####[8]上部サービスライン真ん中
#####[9]上部サービスライン右
#####[10]上部サービスライン左
#####[11]ど真ん中ネット下
#####[12]ど真ん中ネット右
#####[13]ど真ん中ネット左
#####[14]下部サービスライン真ん中
#####[15]下部サービスライン右
#####[16]下部サービスライン左
#####[17]ネット上中
#####[18]ネット上右
#####[19]ネット上左




##############################
'''

def frame_sub(img2, img3, th):
    # フレームの絶対差分
    #diff1 = cv2.absdiff(img1, img2)
    diff = cv2.absdiff(img2, img3)

    # 2つの差分画像の論理積
    #diff = cv2.bitwise_and(diff1, diff2)

    # 二値化処理
    diff[diff < th] = 0
    diff[diff >= th] = 255

    # メディアンフィルタ処理（ゴマ塩ノイズ除去）
    mask = cv2.medianBlur(diff, 3)

    return  mask

def createRotateMat(degree):
    import math
    rad=math.radians(degree)
    return np.array([[math.cos(rad),-math.sin(rad)],[math.sin(rad),math.cos(rad)]])

def hafuline(pts,frame):

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    retval, black2 = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)#画像を2値化
    kernel = np.ones((5,5),np.uint8)
    maxcount=0
    maxcountb=0
    maxcordinate=[]
    maxcordinateb=[]
    flaga=0
    flagb=0

    #cv2.imshow("hafuc", img)
    dst3 = cv2.GaussianBlur(gray, ksize=(3,3), sigmaX=1.3)
    #print(dst3.dtype)

    edges = cv2.Canny(dst3,50,150,apertureSize = 3)

    #gimg = skimage.color.rgb2gray(frame)
    #gray3=edges.astype(np.bool)
    #print(gimg.dtype)
    #gray11=skeletonize(gimg)
    #gray2.astype(np.uint8)
    #gray31=gray11.astype(np.uint8)
    #gray31[gray31 < 1] = 0
    #gray31[gray31 >= 1] = 255
    #gray32=dilation(100,1000,gray31)
    #print(gray31)
    #cv2.imshow("hafua", edges)
    #io.imshow(gray2)
    #retval, gray = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)#画像を2値化
    #erosion = cv2.erode(gray,kernel,iterations = 1)

    #cv2.imshow("hafub", gray2)
    #cv2.imshow("hafud", erosion)
    lines = cv2.HoughLines(edges,1,np.pi,20)
    if(lines is None):
        return
    for i in range(len(lines)):
      for rho,theta in lines[i]:
        #if(theta>=np.pi/2-0.1 and theta<=np.pi/2+0.1):
         #if(theta>=np.pi/2-0.1 or theta<=np.pi/2+0.1):
           #print(theta)
           #frame=cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
           count=0
           a = np.cos(theta)
           b = np.sin(theta)
           x0 = a*rho
           y0 = b*rho
           x1 = int(x0 + 1000*(-b))
           y1 = int(y0 + 1000*(a))
           x2 = int(x0 - 1000*(-b))
           y2 = int(y0 - 1000*(a))
           #frame=cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
           x=int(rho)
           y=int((pts[2][1]-pts[1][1]) * (rho-pts[1][0]) / (pts[2][0]-pts[1][0]) +pts[1][1])
           ya=int((pts[3][1]-pts[0][1]) * (rho-pts[0][0]) / (pts[3][0]-pts[0][0]) +pts[0][1])
           #print(x1,y1,x2,y2,rho,theta)

           if(x>pts[1][0] and x<pts[7][0] and y>pts[9][1] and y < pts[12][1] ):
                 #pts19x=int((pts[2][1]-pts[1][1]) * (rho-pts[1][0]) / (pts[2][0]-pts[1][0]) +pts[1][1])
                 #pts19y=int((pts[17][1]-y) * (i-x) / (pts[17][0]-x) +y)
                 frame = cv2.circle(frame,(x,y), 5, (255,0,255), -1)
                 for i in range(pts[8][0],pts[1][0],1):
                    if(black[int((pts[17][1]-y) * (i-x) / (pts[17][0]-x) +y),i]==255):
                       count+=1
                       #frame = cv2.circle(frame,(i,int((pts[17][1]-y) * (i-x) / (pts[17][0]-x) +y)), 5, (0,255,255), -1)
                 if(count>maxcount):
                     maxcount=count
                     flaga=1
                     maxcordinate=[x,y]
           if(x>pts[5][0] and x<pts[4][0] and ya>pts[10][1] and ya < pts[13][1] ):
                 #frame = cv2.circle(frame,(x,ya), 5, (255,0,255), -1)
                 for i in range(pts[0][0],pts[8][0],1):
                    if(black[int((pts[17][1]-ya) * (i-x) / (pts[17][0]-x) +ya),i]==255):
                       count+=1
                 if(count>maxcountb):
                     maxcountb=count
                     flagb=1
                     maxcordinateb=[x,ya]
                     #print(maxcordinateb)
    #frame = cv2.circle(frame,(maxcordinate[0],maxcordinate[1]), 5, (0,0,255), -1)
    #frame = cv2.circle(frame,(maxcordinateb[0],maxcordinateb[1]), 5, (0,0,255), -1)
    if(flaga==1):
        pts[18]=(maxcordinate[0],maxcordinate[1])
    if(flagb==1):
        pts[19]=(maxcordinateb[0],maxcordinateb[1])
    #frame=cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    return pts[18],pts[19]




    #cv2.imshow("hafu", frame)
def calchomo(x,y,M):

   #x = np.array(x,dtype='float32')
   #y = np.array(y,dtype='float32')
   x=int(x)
   y=int(y)
   x = np.array(x)
   y = np.array(y)

   xx=int((M[0][0]*x+M[0][1]*y+M[0][2])/(M[2][0]*x+M[2][1]*y+M[2][2]))
   yy=int((M[1][0]*x+M[1][1]*y+M[1][2])/(M[2][0]*x+M[2][1]*y+M[2][2]))
   return xx,yy
def dilation(dilationSize,kernelSize,img):#膨張処理。二値化画像を入力に膨張画像を返す。
    kernel=np.ones((kernelSize,kernelSize),np.uint8)#膨張処理のもととなる全要素1の配列の作成。kernelsizeになり大きいほど膨張が大きくなる。
    element=cv2.getStructuringElement(cv2.MORPH_RECT,(2*dilationSize+1,2*dilationSize+1),(dilationSize,dilationSize))#膨張の回数。

    #dilation_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    dilation_img2 = cv2.dilate(img,kernel,element)
    return dilation_img2

def calcnorm(p1,p2,p3):#引:p1,p2が線分上の点、p3が距離を測りたい点。出:距離

    u = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v = np.array([p3[0] - p1[0], p3[1] - p1[1]])
    L = abs(np.cross(u, v) / np.linalg.norm(u))
    return L

def PointOfIntersection(x1,y1,x2,y2,x3,y3,x4,y4):#(x1,y1)(x2,y2)を通る直線と(x3,y3)(x4,y4)を通る直線の交点を求める。

    a1 = (y2-y1)/(x2-x1)
    a3 = (y4-y3)/(x4-x3)
    x = (a1*x1-y1-a3*x3+y3)/(a1-a3)
    y = (y2-y1)/(x2-x1)*(x-x1)+y1
    return x,y

def linechecker(ball_p,pts):#引:ボール座標、コート座標。出;ライン上でなければTrueと座標。ライン上ならFalse


    ###############courtpoint####################
    #####[0]コート左上
    #####[1]コート右上
    #####[2]コート右下
    #####[3]コート左下
    #####[4]上部左シングルコート点
    #####[5]下部左シングルコート点
    #####[6]上部右シングルコート点
    #####[7]下部右シングルコート点
    #####[8]上部サービスライン真ん中
    #####[9]上部サービスライン右
    #####[10]上部サービスライン左
    #####[11]ど真ん中ネット下
    #####[12]ど真ん中ネット右
    #####[13]ど真ん中ネット左
    #####[14]下部サービスライン真ん中
    #####[15]下部サービスライン右
    #####[16]下部サービスライン左
    print(pts)
    distance=10

    '''
    pts_list = pts.tolist()
    #print(pts_list)
    #paa1,paa2,paa3,paa4=[229, 90],[147, 280],[508, 280],[429, 90]#補正前（台形）4点
    ca1,ca2,ca3,ca4=[0, 0],[365, 0],[365, 792],[0, 792]#補正後（正方形）4点
    #ca1,ca2,ca3,ca4=[0, 0],[1097, 0],[1097, 2377],[0, 2377]#補正後（正方形）4点
    src_pts = np.float32([pts[0],pts[1],pts[2],pts[3]]).reshape(-1,1,2)
    dst_pts = np.float32([ca1,ca2,ca3,ca4]).reshape(-1,1,2)
    print(src_pts)
    print(dst_pts)

    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    #print(M)
    dst = cv2.warpPerspective(framecopy,M,(365,792))
    dst = cv2.circle(dst,(182,182), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(45,182), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(320,182), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(182,396), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(45,396), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(320,396), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(182,609), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(45,609), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(320,609), 5, (0,0,255), -1)
    cv2.imshow("mask", dst)



    src_pts = np.float32([ca1,ca2,ca3,ca4]).reshape(-1,1,2)
    dst_pts = np.float32([pts[0],pts[1],pts[2],pts[3]]).reshape(-1,1,2)
    M2 = cv2.getPerspectiveTransform(src_pts,dst_pts)
    dst2 = cv2.warpPerspective(dst,M2,(1000,1500))
    cv2.imshow("mask2", dst2)

    e=calchomo(548.5/3,548.5/3,M2)#テニスコート奥の中点
    print(e)
    #print(type(M2))
    f=calchomo(960/3,548.5/3,M2)#テニスコート奥の右サービスライン点
    g=calchomo(137/3,548.5/3,M2)#テニスコート奥の左サービスライン点
    h=calchomo(548.5/3,1188.5/3,M2)#テニスコート奥の中点
    i=calchomo(960/3,1188.5/3,M2)#テニスコート奥の中点

    j=calchomo(137/3,1188.5/3,M2)#テニスコート奥の中点
    k=calchomo(548.5/3,1828.5/3,M2)#テニスコート奥の中点
    l=calchomo(960/3,1828.5/3,M2)#テニスコート奥の中点
    m=calchomo(137/3,1828.5/3,M2)#テニスコート奥の中点


    courtpoint = pts.copy()
    count=0
    #print(type(courtpoint))
    a=[int(0.876*pts[0][0]+0.124*pts[1][0]),int(0.876*pts[0][1]+0.124*pts[1][1])]#上部左シングルコート点
    b=[int(0.876*pts[3][0]+0.124*pts[2][0]),int(0.876*pts[3][1]+0.124*pts[2][1])]#下部左シングルコート点
    c=[int(0.124*pts[0][0]+0.876*pts[1][0]),int(0.124*pts[0][1]+0.876*pts[1][1])]#上部右シングルコート点
    d=[int(0.124*pts[3][0]+0.876*pts[2][0]),int(0.124*pts[3][1]+0.876*pts[2][1])]#下部右シングルコート点
    #print(a)
    courtpoint=np.vstack((courtpoint,a))
    courtpoint=np.vstack((courtpoint,b))
    courtpoint=np.vstack((courtpoint,c))
    courtpoint=np.vstack((courtpoint,d))
    courtpoint=np.vstack((courtpoint,e))
    courtpoint=np.vstack((courtpoint,f))
    courtpoint=np.vstack((courtpoint,g))
    courtpoint=np.vstack((courtpoint,h))
    courtpoint=np.vstack((courtpoint,i))
    courtpoint=np.vstack((courtpoint,j))
    courtpoint=np.vstack((courtpoint,k))
    courtpoint=np.vstack((courtpoint,l))
    courtpoint=np.vstack((courtpoint,m))



    #np.insert(courtpoint,0.5)
    #courtpoint.append([0.124*(pts[3][0]+pts[4][0]),0.124*(pts[3][1]+pts[4][1])])
    #courtpoint =np.append(courtpoint,[2,3],axis=0)
    '''
    for ii in range(len(pts)):
         #np.insert(pts,0,pts[i])
         if(calcnorm(pts[0],pts[1],ball_p)<distance):#上のベースライン
             return False
         if(calcnorm(pts[1],pts[2],ball_p)<distance):#右のダブルスライン
             return False
         if(calcnorm(pts[2],pts[3],ball_p)<distance):#下のベースライン
             return False
         if(calcnorm(pts[0],pts[3],ball_p)<distance):#左のダブルスライン
             return False
         if(calcnorm(pts[4],pts[5],ball_p)<distance):#左シングルライン
             return False
         if(calcnorm(pts[6],pts[7],ball_p)<distance):#右シングルライン
             return False
         if(calcnorm(pts[10],pts[8],ball_p)<distance):#upservicelineleft
             return False
         if(calcnorm(pts[8],pts[9],ball_p)<distance):#upservicelineright
             return False
         if(calcnorm(pts[8],pts[14],ball_p)<distance):#centerline
             return False
         if(calcnorm(pts[15],pts[16],ball_p)<distance):#downserviceline
             return False
         if (calcnorm(pts[17],pts[18],ball_p)<distance):#downserviceline
             return False
         if (calcnorm(pts[13],pts[12],ball_p)<distance):#downserviceline
             return False
         if (calcnorm(pts[17],pts[19],ball_p)<distance):#downserviceline
             return False
         if(ball_p[1] < (pts[7][1]-pts[6][1]) * (ball_p[0]-pts[6][0]) / (pts[7][0]-pts[6][0]) +pts[6][1]):#downserviceline
             return False
         if(ball_p[1] < (pts[4][1]-pts[5][1]) * (ball_p[0]-pts[5][0]) / (pts[4][0]-pts[5][0]) +pts[5][1]):#downserviceline
             return False
         if(ball_p[1] > (pts[2][1]-pts[3][1]) * (ball_p[0]-pts[3][0]) / (pts[2][0]-pts[3][0]) +pts[3][1]):#downserviceline
             return False
         if(ball_p[1] < (pts[0][1]-pts[1][1]) * (ball_p[0]-pts[1][0]) / (pts[0][0]-pts[1][0]) +pts[1][1]):
             return False
         return True
    #courtpoint.([1,2])
    #print(courtpoint[4])
    #print(courtpoint)
    #for i in courtpoint
    #frame = cv2.circle(img,(courtpoint[i][0],approx[i][1]), 5, (0,0,255), -1)

def person1detect(img):#カラー（未処理）画像を入力に、赤、白、黒のピクセル数を求め面積で割り割合を返す。

    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 赤色のHSVの値域1
    hsv_min = np.array([0,127,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,127,0])
    hsv_max = np.array([179,255,255])

    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
    mask3 = mask1+mask2
    hist = cv2.calcHist([mask3], [0], None, [256], [0, 256])#バウンディングボックスに取得した背景と人物からマスクで切り取った人物の部分の画像を抽出しヒストグラムを出力。


    ##whitedetect
    retval, white = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)#画像を2値化
    retval2, black = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)#画像を2値
    #dst = cv2.bitwise_or(white, black)
    #cv2.imshow("black",black)
    #hist = cv2.calcHist([white], [0], None, [256], [0, 256])#バウンディングボックスに取得した背景と人物からマスクで
    histw = cv2.calcHist([white], [0], None, [256], [0, 256])#バウンディングボックスに取得した背景と人物からマスクで切り取った人物の部分の画像を抽出しヒストグラムを出力。
    histb = cv2.calcHist([black], [0], None, [256], [0, 256])#バウンディングボックスに取得した背景と人物からマスクで切り取った人物の部分の画像を抽出しヒストグラムを出力。
    #print(img.shape[0])
    #print(img.shape[1])
    sum=(100*hist[255]+histw[255])/(img.shape[1]*img.shape[0])
    #print(sum)
    #hsv_minw = np.array([0,0,100])
    #sv_maxw = np.array([179,45,255])
    #maskw = cv2.inRange(hsv, hsv_minw, hsv_maxw)

    #histw = cv2.calcHist([maskw], [0], None, [256], [0, 256])#バウンディングボックスに取得した背景と人物からマスクで切り取った人物の部分の画像を抽出しヒストグラムを出力。

    #cv2.imshow("Tracking",mask3)
    #k = cv2.waitKey(0) & 0xff

    #plt.xlim(0, 255)
    #plt.plot(hist)
    #plt.xlabel("Pixel value", fontsize=20)
    #plt.ylabel("Number of pixels", fontsize=20)
    #plt.grid()
    #plt.show()
    #k = cv2.waitKey(0) & 0xff

    return sum

def court_track(img,pts,frame):#グレー画像,前回のコートの角位置を入力に白色で囲まれた最大面積のものを抽出し黒で囲む。
    #gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#グレー画像に変
    count=0
    #pts17maxcount=0
    imgcopy = img.copy()
    framecopy=frame.copy()
    retval, black2 = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)#画像を2値化
    #black = dilation(10,10,black)
    #black2 = dilation(10,5,black)
    cv2.imshow("black",black2)

    ###################if文によるコートの四隅の判定##########################
    ####４行一列の配列の中身が2変数。つまり4個のデータがあり、中身がx,y座標のようなデータは4*2の配列で表される。
    ######画像はy軸、x軸で表される・
    ######pts[左上、右上、右下、左下]
    ###
    #s=0.5*
    print(pts)
    if(pts[1][0]!=0):

      for i in range(len(pts)):

         if(black[pts[i][1],pts[i][0]]==255):#左上検出	サイズ	名前	日時
            count+=1

      if(count>=14 and cv2.arcLength(pts,True)>2000):#左下検出 xが一番小さい

        cv2.polylines(frame,[pts[0:4]],True,(0,255,0))
        pts_list = pts.tolist()
        for i in range(len(pts)):
            frame = cv2.circle(frame,(pts[i][0],pts[i][1]), 5, (255,0,255), -1)
        #print(cv2.arcLength(pts,True))
        '''
        cv2.polylines(frame,[pts],True,(0,255,0))
        frame = cv2.circle(frame,(pts[0][0],pts[0][1]), 5, (0,0,255), -1)
        frame = cv2.circle(frame,(pts[1][0],pts[1][1]), 5, (0,0,255), -1)
        frame = cv2.circle(frame,(pts[2][0],pts[2][1]), 5, (0,0,255), -1)
        frame = cv2.circle(frame,(pts[3][0],pts[3][1]), 5, (0,0,255), -1)
        frame = cv2.circle(frame,(int(0.5*(pts[3][0]+pts[2][0])),pts[3][1]), 5, (0,0,255), -1)
        frame = cv2.circle(frame,(int(0.5*(pts[1][0]+pts[0][0])),pts[0][1]), 5, (0,0,255), -1)
        frame = cv2.circle(frame,(int(0.5*(pts[0][0]+pts[3][0])),int(0.5*(pts[3][1]+pts[0][1]))), 5, (0,0,255), -1)
        frame = cv2.circle(frame,(int(0.5*(pts[2][0]+pts[1][0])),int(0.5*(pts[1][1]+pts[2][1]))), 5, (0,0,255), -1)
        #frame = cv2.circle(frame,(int(0.876*pts[0][0]+0.124*pts[1][0]),int(0.876*pts[0][1]+0.124*pts[1][1])), 5, (0,0,255), -1)
        #frame = cv2.circle(frame,(int(0.876*pts[3][0]+0.124*pts[2][0]),int(0.876*pts[3][1]+0.124*pts[2][1])), 5, (0,0,255), -1)
        '''
        #cv2.imshow("frame",frame)
        #print("keep")
        return pts,frame

	###################if文によるコートの四隅の判定##########################


    image,contours, hierarchy = cv2.findContours(black2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#輪郭の検出
    #輪郭の中で面積が最大となる輪郭を検出
    #print(contours)
    max_area=0
    tempcount=0
    for c in contours:
        area = cv2.contourArea(c)#面積計算
        if(area>max_area):
            max_area=area
            temp=c
            tempcount+=1
    #近似



    if(tempcount==0):
       return pts,frame

    epsilon = 0.005 * cv2.arcLength(temp, True)#輪郭の周囲長
    approx = cv2.approxPolyDP(temp, epsilon, True)#輪郭の近似
    img2=cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)#輪郭を描写
    #print(temp)


    for i in range(len(approx)):
        img = cv2.circle(img,(approx[i][0][0],approx[i][0][1]), 5, (0,0,255), -1)

    cv2.imshow("img",img)


    ###############コートの検出########################

    rotate1=createRotateMat(135)#左上検出用の回転マトリクス
    rotate2=createRotateMat(45)#右上検出用の回転マトリクス

    #初期座標
    min_x_point=approx[0][0]#座標「○、○」
    max_x_point=approx[0][0]
    left_max_point=approx[0][0]
    right_max_point=approx[0][0]
    left_max_point_temp=np.array(approx[0][0])
    left_max_point_rot=rotate1.dot(left_max_point_temp)
    right_max_point_temp=np.array(approx[0][0])
    right_max_point_rot=rotate2.dot(right_max_point_temp)

    #近似図形の全部の点を参照して4隅の点を検出
    for i in range(len(approx)):
            x=approx[i][0][0]#xは点のx座標
            y=approx[i][0][1]

            #if(min_x_point[1]<y or max_x_point[1]<y):#左下検出 xが一番小さい
            if(min_x_point[0]>x):#左下検出 xが一番小さい
                   min_x_point=[x,y]
            if(max_x_point[0]<x):#右下検出 xが一番大きい
                   max_x_point=[x,y]
            if(x<=640 and y<360):
                if(left_max_point[1]-10 > y):
                    left_max_point=[x,y]
                if(left_max_point[1]-10 <= y <=left_max_point[1]+10):#右下検出 xが一番大きい
                    if(x<left_max_point[0]):
                        left_max_point=[x,y]
            if(x>=640 and y<360):
               if(right_max_point[1]-10 > y):
                    right_max_point=[x,y]
               if(right_max_point[1]-10 <= y <=right_max_point[1]+10):#右下検出 xが一番大きい
                    if(x>right_max_point[0]):
                        right_max_point=[x,y]
                #if(right_max_point[0] <= x):

            '''
            point=np.array([x,y])
            rotate_point1=rotate1.dot(point)
            rotate_point2=rotate2.dot(point)

            if(left_max_point_rot[0]<rotate_point1[0]):#左上検出 回転後のxが一番大きい
                left_max_point_rot=rotate_point1
                left_max_point=[x,y]
            if(right_max_point_rot[0]<rotate_point2[0]):#右上検出 回転後のxが一番小さい
                right_max_point_rot=rotate_point2
                right_max_point=[x,y]
            '''

    point3=np.array(min_x_point)#左下
    point4=np.array(max_x_point)#右下
    point1=np.array(left_max_point)#左上
    point2=np.array(right_max_point)#右上

    ###############courtpoint####################
    #####[0]コート左上
    #####[1]コート右上
    #####[2]コート右下
    #####[3]コート左下
    #####[4]上部左シングルコート点
    #####[5]下部左シングルコート点
    #####[6]上部右シングルコート点
    #####[7]下部右シングルコート点
    #####[8]上部サービスライン真ん中
    #####[9]上部サービスライン右
    #####[10]上部サービスライン左
    #####[11]ど真ん中ネット下
    #####[12]ど真ん中ネット右
    #####[13]ど真ん中ネット左
    #####[14]下部サービスライン真ん中
    #####[15]下部サービスライン右
    #####[16]下部サービスライン左
    #####[17]ネット上中
    #####[18]ネット上右
    #####[19]ネット上左

    #print(pts_list)
    #paa1,paa2,paa3,paa4=[229, 90],[147, 280],[508, 280],[429, 90]#補正前（台形）4点
    ca1,ca2,ca3,ca4=[0, 0],[365, 0],[365, 792],[0, 792]#補正後（正方形）4点
    pts[0]=point1
    pts[1]=point2
    pts[2]=point4
    pts[3]=point3
    print(pts)
    #ca1,ca2,ca3,ca4=[0, 0],[1097, 0],[1097, 2377],[0, 2377]#補正後（正方形）4点
    src_pts = np.float32([pts[0],pts[1],pts[2],pts[3]]).reshape(-1,1,2)
    dst_pts = np.float32([ca1,ca2,ca3,ca4]).reshape(-1,1,2)


    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    #print(M)
    dst = cv2.warpPerspective(framecopy,M,(365,792))
    dst = cv2.circle(dst,(182,182), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(45,182), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(320,182), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(182,396), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(45,396), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(320,396), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(182,609), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(45,609), 5, (0,0,255), -1)
    dst = cv2.circle(dst,(320,609), 5, (0,0,255), -1)
    cv2.imshow("mask", dst)



    src_pts = np.float32([ca1,ca2,ca3,ca4]).reshape(-1,1,2)
    dst_pts = np.float32([pts[0],pts[1],pts[2],pts[3]]).reshape(-1,1,2)
    M2 = cv2.getPerspectiveTransform(src_pts,dst_pts)
    dst2 = cv2.warpPerspective(dst,M2,(1000,1500))
    cv2.imshow("mask2", dst2)

    pts[8]=calchomo(548.5/3,548.5/3,M2)#テニスコート奥の中点
    #print(e)
    #print(type(M2))
    pts[9]=calchomo(960/3,548.5/3,M2)#テニスコート奥の右サービスライン点
    pts[10]=calchomo(137/3,548.5/3,M2)#テニスコート奥の左サービスライン点
    pts[11]=calchomo(548.5/3,1188.5/3,M2)#テニスコート奥の中点
    pts[12]=calchomo(960.0/3,1188.5/3,M2)#テニスコート奥の中点
    #print(pts[12])

    pts[13]=calchomo(137/3,1188.5/3,M2)#テニスコート奥の中点
    pts[14]=calchomo(548.5/3,1828.5/3,M2)#テニスコート奥の中点
    pts[15]=calchomo(960/3,1828.5/3,M2)#テニスコート奥の中点
    pts[16]=calchomo(137/3,1828.5/3,M2)#テニスコート奥の中点
    #pts[17]=calchomo(548.5/3,1097.1/3,M2)#テニスコート奥の中点
    #pts[18]=calchomo(960/3,1081.5/3,M2)#テニスコート奥の中点
    #pts[19]=calchomo(137/3,1081.5/3,M2)#テニスコート奥の中点

    distance=10
    #courtpoint = pts.copy()
    #ount=0
    #print(type(courtpoint))
    pts[4]=[int(0.876*pts[0][0]+0.124*pts[1][0]),int(0.876*pts[0][1]+0.124*pts[1][1])]#上部左シングルコート点
    pts[5]=[int(0.876*pts[3][0]+0.124*pts[2][0]),int(0.876*pts[3][1]+0.124*pts[2][1])]#下部左シングルコート点
    pts[6]=[int(0.124*pts[0][0]+0.876*pts[1][0]),int(0.124*pts[0][1]+0.876*pts[1][1])]#上部右シングルコート点
    pts[7]=[int(0.124*pts[3][0]+0.876*pts[2][0]),int(0.124*pts[3][1]+0.876*pts[2][1])]#下部右シングルコート点

    max17cordinate=[]
    pts17maxcount=0
    flag=0
    for i in range(pts[8][1]+5,pts[11][1]-5,1):
        pts17count=0
        for j in range(-10,10,1):
                    if(black[i,pts[8][0]+j]==255):
                       pts17count+=1
        if(pts17count>pts17maxcount):
                     pts17maxcount=pts17count
                     max17cordinate=[pts[8][0],i]
                     flag=1
    if(flag==1):
        pts[17]=(max17cordinate[0],max17cordinate[1])
    pts[18],pts[19]=hafuline(pts,framecopy)
    #print(pts[17])


    '''
    pts=np.vstack((pts,a))
    pts=np.vstack((pts,b))
    pts=np.vstack((pts,c))
    pts=np.vstack((pts,d))
    pts=np.vstack((pts,e))
    pts=np.vstack((pts,f))
    pts=np.vstack((pts,g))
    pts=np.vstack((pts,h))
    pts=np.vstack((pts,i))
    pts=np.vstack((pts,j))
    pts=np.vstack((pts,k))
    pts=np.vstack((pts,l))
    pts=np.vstack((pts,m))
    '''
    #print(pts)






    #cv2.imshow("img",img)
    #pts=np.array([point1,point2,point4,point3],dtype=int)#pts[左上、右上、右下、左下]
    #pts[左上、右上、右下、左下]
    print(pts[0:4])
    cv2.polylines(frame,[pts[0:4]],True,(0,255,0))
    pts_list = pts.tolist()
    for i in range(len(pts)):
        frame = cv2.circle(frame,(pts[i][0],pts[i][1]), 5, (0,0,255), -1)
    #cv2.imshow("frame",frame)
    ##################################################

    return pts,frame


def red_detect(img):#カラー（未処理）画像を入力に赤色のピクセル数を返す。
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([0,127,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,127,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
    mask3 = mask1+mask2
    hist = cv2.calcHist([mask3], [0], None, [256], [0, 256])#バウンディングボックスに取得した背景と人物からマスクで切り取った人物の部分の画像を抽出しヒストグラムを出力。

    #cv2.imshow("Tracking",mask3)
    #k = cv2.waitKey(0) & 0xff

    #plt.xlim(0, 255)
    #plt.plot(hist)
    #plt.xlabel("Pixel value", fontsize=20)
    #plt.ylabel("Number of pixels", fontsize=20)
    #plt.grid()
    #plt.show()
    #k = cv2.waitKey(0) & 0xff
    sum=(hist[255])/(img.shape[1]*img.shape[0])
    #print(sum)
    return sum

def white_detect(img):#カラー（未処理）画像を入力に白色のピクセル数を返す。
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([0,0,100])
    hsv_max = np.array([179,45,255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    hist = cv2.calcHist([mask], [0], None, [256], [0, 256])#バウンディングボックスに取得した背景と人物からマスクで切り取った人物の部分の画像を抽出しヒストグラムを出力。

    #cv2.imshow("Tracking",mask3)
    #k = cv2.waitKey(0) & 0xff

    #plt.xlim(0, 255)
    #plt.plot(hist)
    #plt.xlabel("Pixel value", fontsize=20)
    #plt.ylabel("Number of pixels", fontsize=20)
    #plt.grid()
    #plt.show()
    #k = cv2.waitKey(0) & 0xff

    return hist[255]

def yellow_detect(img):#カラー（未処理）画像を入力に黃色のピクセル数を返す。
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([20,127,0])
    hsv_max = np.array([35,255,255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    hist = cv2.calcHist([mask], [0], None, [256], [0, 256])#バウンディングボックスに取得した背景と人物からマスクで切り取った人物の部分の画像を抽出しヒストグラムを出力。

    #cv2.imshow("Tracking",mask)
    #k = cv2.waitKey(0) & 0xff

    #plt.xlim(0, 255)
    #plt.plot(hist)
    #plt.xlabel("Pixel value", fontsize=20)
    #plt.ylabel("Number of pixels", fontsize=20)
    #plt.grid()
    #plt.show()
    #k = cv2.waitKey(0) & 0xff

    return hist[255]

def flesh_detect(img):#カラー（未処理）画像を入力に肌色のピクセル数を返す。
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([0,58,88])
    hsv_max = np.array([25,173,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,58,88])
    hsv_max = np.array([179,173,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
    mask3 = mask1+mask2
    hist = cv2.calcHist([mask3], [0], None, [256], [0, 256])#バウンディングボックスに取得した背景と人物からマスクで切り取った人物の部分の画像を抽出しヒストグラムを出力。

    #cv2.imshow("Tracking",mask3)
    #k = cv2.waitKey(0) & 0xff

    #plt.xlim(0, 255)
    #plt.plot(hist)
    #plt.xlabel("Pixel value", fontsize=20)
    #plt.ylabel("Number of pixels", fontsize=20)
    #plt.grid()
    #plt.show()
    #k = cv2.waitKey(0) & 0xff

    return hist[255]

def mouse_event(event, x, y, flag, params):
    if event == cv2.EVENT_LBUTTONDOWN:
            #print(x,y)
            params=[x, y]
            #print(params)

        # カーソルの座標を取得（2点目）
    elif event == cv2.EVENT_RBUTTONDOWN:
            self.point2 = (x, y)
            print('point2: = ', self.point2)

    elif event == cv2.EVENT_MBUTTONDOWN:
            print('M:')
            print('point1: = ', self.point1)
            print('point2: = ', self.point2)
            cv2.waitKey(0)



def ball_track(frame,mask,pts,bbox,bbox2,recentball,motion3):#frame二値化画像、mask膨張画像
    #def ball_track(height,width):#動体の膨張画像を基に、人間、
    if(pts[0][0]==0.0):
        return frame,recentball,motion3
    b=np.array(recentball)
    #print(b)
    distmin=10000
    distpos=[]
    #cv2.imshow("frame", frame)
    mask1 = mask.copy()
    counter=0
    currentball_pos=[]
    #print(pts)
    #print(bbox2[0],bbox2[1],bbox2[2],bbox2[3])

    if(bbox[0]!=0 and bbox[1]!=0 and bbox[2]!=0 and bbox[3]!=0):
         if(bbox2[0]!=0 and bbox2[1]!=0 and bbox2[2]!=0 and bbox2[3]!=0):
               imageArray = np.zeros((int(mask1.shape[0]),int(mask1.shape[1])),np.uint8)#動画と同サイズの黒画像の作成
               imageArray2 = np.zeros((int(h1),int(w1)),np.uint8)#動画と同サイズの黒画像の作成
               imageArray3 = np.zeros((int(h2),int(w2)),np.uint8)#動画と同サイズの黒画像の作成
               #cv2.imshow("ima", imageArray2)
               mask1[r1:r1+h1,c1:c1+w1]=imageArray2
               mask1[r2:r2+h2,c2:c2+w2]=imageArray3


               #cv2.imshow("imageArray", mask1)
               if(pts[0][1]!=0 and pts[3][1]!=0 and pts[3][0]!=0 and pts[2][0]!=0):
                    roi = mask1[pts[0][1]:pts[3][1],pts[3][0]:pts[2][0]]
                    if(len(roi)!=0):
                        #cv2.imshow("imageArray", roi)
                        roi2 = np.zeros((int(roi.shape[0]),int(roi.shape[1])),np.uint8)#動画と同サイズの黒画像の作成
                        #image,contours, hierarchy = cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#輪郭の検出
                        image,contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#輪郭の検出
                        #輪郭の中で面積が最大となる輪郭を検出
                        #print(contours)
                        min_area=1000000


                        for i in range(len(contours)):  # 重心位置を計算
                           count = len(contours[i])
                           area = cv2.contourArea(contours[i])  # 面積計算
                           if(area<1000):
                               x, y = 0.0, 0.0
                               for j in range(count):
                                   x += contours[i][j][0][0]
                                   y += contours[i][j][0][1]
                               x /= count
                               y /= count
                               x = int(x)
                               y = int(y)
                               ball=[x,y]
                               #print(x,y)
                               #####pts[左上、右上、右下、左下]
                               #if(x>=pts[3][0] and y>=pts[0][1] and x< pts[2][0] and y<=pts[2][1]):
                               if(linechecker(ball,pts)==True):
                                         #print(x,y)
                                         currentball_pos.append(ball)
                                         counter+=1

                    """
                    for c in contours:
                        area = cv2.contourArea(c)#面積計算
                        if(area<min_area):
                            min_area=area
                            #temp=area
                            print(area)
                            #temp.append([c[0][0],c[0][1]])
                            #print(temp)

                    """

                    #print(currentball_pos)
                    if(counter==0):
                        print("excape")
                        return frame,recentball,motion3

                    #plt.xlabel("x")
                    #plt.ylabel("y")
                    #plt.title('sin & cos')

                    #epsilon = 0.005 * cv2.arcLength(temp, True)#輪郭の周囲長
                    #approx = cv2.approxPolyDP(temp, epsilon, True)#輪郭の近似
                    for i in range(len(currentball_pos)):#近傍探索
                        a=np.array(currentball_pos[i])#a全体点、b近傍点
                        #cv2.rectangle(frame, (currentball_pos[i][0], currentball_pos[i][1]), (0,255,0),2)#描画関数赤。p全仏1が短形の一つの頂点、p2がもうひとつの頂点で
                        dist = np.linalg.norm(a-b)
                        if(distmin>dist):
                            distmin=dist
                            distpos=i
                            recentball=currentball_pos[i]
                        #print(dist)
                        #roi3 = cv2.circle(roi2,(int(roi.shape[0])+ball_pos[i][0],ball_pos[i][1]+int(roi.shape[1])), 5, (255,255,255), -1)
                        frame = cv2.circle(frame,(currentball_pos[i][0],currentball_pos[i][1]), 5, (255,255,255), -1)
                        #linechecker(recentball,pts)
                        #plt.plot(ball_pos[i][0], ball_pos[i][1],'o')


                    #plt.legend()
                    # グラフの描画実行
                    #plt.show()
               #cv2.imshow("roi", frame2)
               print(recentball[0],recentball[1])
               p1 = (recentball[0]-10, recentball[1]-10)#x,y座標
               p2 = (recentball[0] + 10, recentball[1] + 10)#x座標+横幅、y座標+縦幅
               g3=(int(recentball[0]-10)+int(0.5*(20)), int(recentball[1]-10)+int(0.5*(20)))
               motion3.append(g3)
               for i in range(len(motion3)-1):
                    cv2.line(frame,motion3[i],motion3[i+1],(255,255,0),2)
               cv2.rectangle(frame, p1, p2, (0,255,0),2)#描画関数赤。p全仏1が短形の一つの頂点、p2がもうひとつの頂点で
               cv2.putText(frame, "Tracking_ball", (recentball[0], recentball[1]), fontType, 1, (0, 255, 0), 1,)
               frame2 = cv2.circle(frame,(recentball[0],recentball[1]), 5, (255,0,0), -1)
               #linechecker(recentball,pts)
               return frame,recentball,motion3

               '''
               if(len(roi3)!=0):
                        #print(pts[0][1],pts[3][1],pts[3][0],pts[2][0])
                        cv2.imshow("roi", roi3)
                        return ball_pos
               if(len(roi3)==0):
                        return ball_pos
               '''


#def nearball_track(ball_p,mask):






#トラッキングタイプでMILを選択
#KCF最強TLD学習
tracker= cv2.TrackerKCF_create()
tracker2= cv2.TrackerKCF_create()
tracker3= cv2.TrackerKCF_create()

#hog特徴検出機の準備
hog = cv2.HOGDescriptor()
hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}


#videoファイルを読み込む
video = cv2.VideoCapture(usevideo)

# ファイルがオープンできない場合の処理.
if not video.isOpened():
    print ("Could not open video")
    sys.exit()
# 最初のフレームを読み込む
okf, frame = video.read()
if not okf:
    print ('Cannot read video file')
    sys.exit()



#ボタン押すまで画像表示
cv2.imshow("framecopy",frame)
k = cv2.waitKey(0) & 0xff


# バウンディングボックスの最初の位置とサイズを設定
#c,r,w,h = 550,330,15,15
c1,r1,w1,h1 = 485,127,50,70
c2,r2,w2,h2 = 690,410,100,150
c3,r3,w3,h3 = 726,229,20,20

bbox = (c1,r1,w1,h1)#x座標、y座標、横幅、縦幅をbboxに格納
bbox2 = (c2,r2,w2,h2)#x座標、y座標、横幅、縦幅をbboxに格納
bbox3 = (c3,r3,w3,h3)#x座標、y座標、横幅、縦幅をbboxに格納

motion1=[]
motion2=[]
motion3=[]

roi = frame[r1:r1+h1, c1:c1+w1]#バウンディングボックスの初期位置からそれぞれの範囲を切り取り。つまりバウンディングボックス分切り取り

point1=[0,0]
point2=[0,0]
point3=[0,0]
point4=[0,0]

pts=np.zeros((20,2),dtype=int)#pts[左上、右上、左下、右下]

ball_pos=[]
recentball=[500,500]
#cv2.setMouseCallback("framecopy", mouse_event)
#print(red_detect(roi))

#min_val,max_val

#cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)#roi_histを正規化してコントラスト低減
#term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)#k-mean法の基準作成。色数を減らす。|はorでepsかcountが80,1のどちらかであれば


# 動画の読み込みと動画情報の取得
#movie = cv2.VideoCapture(target)
fps    = video.get(cv2.CAP_PROP_FPS)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
# 形式はMP4Vを指定
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
result = "result/test_output.m4v"
# 出力先のファイルを開く
out = cv2.VideoWriter('output.avi',fourcc,20.0, (int(width), int(height)))

fontType = cv2.FONT_HERSHEY_SIMPLEX


# バウンディングボックスをフレームに設定
ok = tracker.init(frame, bbox)#入力のbbを追跡開始。バウンディングボックスの座標とフレームを入力、出力に成功を返す。
ok2 = tracker2.init(frame, bbox2)
#print(ok)
#print(type(bbox))

#フレーム差分用の画像所得
#frame1 = cv2.cvtColor(video.read()[1], cv2.COLOR_RGB2GRAY)
frame2 = cv2.cvtColor(video.read()[1], cv2.COLOR_RGB2GRAY)
frame3 = cv2.cvtColor(video.read()[1], cv2.COLOR_RGB2GRAY)





while True:
    # フレームを読み込む
    okf, frame = video.read()
    if not okf:
        print("error")
        break
    framecopy = frame.copy()
    framecopy2=frame.copy()
    #print(recentball)
    #timer = cv2.getTickCount()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval, black = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)#画像を2値化
    #mask = frame_sub(frame1, frame2, frame3, th=30)#フレームの差分から二値化
    mask = frame_sub(frame2, frame3, th=10)#フレームの差分から二値化
    maska=frame_sub(frame2,frame3,th=100)

    #cv2.imshow("mask", mask)
    #cv2.imshow("mask2", maska)
    #mask2 = dilation(1000,10,mask)#第一引数は第二引数が膨張サイズ。
    mask2 = dilation(10,10,mask)#第一引数は第二引数が膨張サイズ。
    maska2 = dilation(10,20,maska)
    black2 = dilation(10,20,black)
    #cv2.imshow("mask", mask2)
    #cv2.imshow("mask2", maska2)
    #cv2.imshow("black",black2)
    #mask2=mask2-black2
    #cv2.imshow("imagefasdfaa", mask2)
    pts,framecopy=court_track(frame2,pts,framecopy)
    #pts,framecopy=court_track(mask2,pts,framecopy)
    #print(framecopy,mask2,pts,bbox,bbox2,recentball,motion3)
    framecopy,recentball,motion3=ball_track(framecopy,mask2,pts,bbox,bbox2,recentball,motion3)
    #hafuline(pts,framecopy2)


    #display_cv_image(dst)
    #M,maskaaaa=cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    #dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
    #track, bbox = tracker.update(frame)
    #track, bbox2 = tracker2.update(frame)
    # フレームを更新
    ok, bbox = tracker.update(frame)#frameを入力、成功とbbのざひょうを返す
    ok2, bbox2 = tracker2.update(frame)
    #ok3, bbox2 = tracker3.update(frame)



    # コート上部の選手のバウンディングボックスを描画
    if ok==True and red_detect(frame[r1:r1+h1, c1:c1+w1])>0.00001: #and b特定の色 ガゾ数box[0] > 300 and bbox[1] > 50:
        #bbox=np.array(bbox)
        #person=frame[r1:r1+h1, c1:c1+w1]
        #if person1detect(person) < 0.5:
            #print("sedfjfjasfl")
            #break
        #print (bbox)
        #roi = frame[y:y+h, x:x+w]
        #if(roi==)
        p1 = (int(bbox[0]), int(bbox[1]))#x,y座標
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))#x座標+横幅、y座標+縦幅
        g1=(int(bbox[0])+int(0.5*bbox[2]), int(bbox[1])+int(0.5*bbox[3]))
        motion1.append(g1)
        for i in range(len(motion1)-1):
            cv2.line(framecopy,motion1[i],motion1[i+1],(255,255,0),2)

        #cv2.rectangle(framecopy, p1, p2, (0,0,255),2)#描画関数赤。p全仏1が短形の一つの頂点、p2がもうひとつの頂点で
        #cv2.putText(framecopy, "Tracking_P1", (int(bbox[0]), int(bbox[1])), fontType, 1, (0, 0, 255), 1,)
        ok, bbox = tracker.update(frame)#frameを入力、成功とbbのざひょうを返す



    else:
      #print(ok)
       #frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
       #frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
       #frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
       #mask = frame_sub(frame1, frame2, frame3, th=30)


      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      human, r = hog.detectMultiScale(gray, **hogParams)
      maxredcount=0
      redcount=0
      maxx=0
      maxy=0
      maxh=0
      maxw=0

      for (x, y, w, h) in human:#humanという配列
         if y > 30 and y < 300 and x > 400 and x < 900:#コート上部付近の人間
             roi = frame[y:y+h, x:x+w]#バウンディングボックスの初期位置からそれぞれの範囲を切り取り。つまりバウンディングボックス分切り取り
             redcount=person1detect(roi)
             if redcount>maxredcount:
                 maxredcount=redcount
                 c1=x
                 r1=y
                 h1=h
                 w1=w

      #print(red_detect(frame[maxy:maxy+maxh, maxx:maxx+maxw]))
      cv2.rectangle(framecopy, (c1, r1), (c1 +w1, r1+h1), (200,0,0),2)#blue
      cv2.putText(framecopy, "Detecting_P1", (c1, r1), fontType, 1, (0, 0, 255), 1)

      bbox= [c1,r1,w1,h1]#x座標、y座標、横幅、縦幅をbboxに格納
      bbox= [float(i) for i in bbox]
      bbox=tuple(bbox)
      #print (bbox)
      tracker.clear();
      tracker= cv2.TrackerKCF_create()
      ok = tracker.init(frame,bbox)
      #print(bbox)
      #print(ok)
      #ok, bbox = tracker.update(frame)#frameを入力、成功とbbのざひょうを返す




    #コート下部の選手のバウンディングボックスを描画
    if ok2==True and bbox2[0] > 300 and bbox2[1] > 100:
        pa1 = (int(bbox2[0]), int(bbox2[1]))#x,y座標
        pa2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))#x座標+横幅、y座標+縦幅
        g2=(int(bbox2[0])+int(0.5*bbox2[2]), int(bbox2[1])+int(0.5*bbox2[3]))
        motion2.append(g2)
        for i in range(len(motion2)-1):
            cv2.line(framecopy,motion2[i],motion2[i+1],(255,255,0),2)
        cv2.rectangle(framecopy, pa1, pa2, (0,0,255),2)#描画関数。p1が短形の一つの頂点、p2がもうひとつの頂点でred
        cv2.putText(framecopy, "Tracking_P2", (int(bbox2[0]), int(bbox2[1])), fontType, 1, (0, 0, 255), 1,)
        ok2, bbox2 = tracker2.update(frame)
    else:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      human2, a2 = hog.detectMultiScale(gray, **hogParams)
      maxwhitecount=0
      whitecount=0
      c2=0
      r2=0
      h2=0
      w2=0

      for (x, y, w, h) in human2:#humanという配列
          if y < 600 and y > 300 and x > 220 and x < 1100:
              roi = frame[y:y+h, x:x+w]#バウンディングボックスの初期位置からそれぞれの範囲を切り取り。つまりバウンディングボックス分切り取り
              whitecount=white_detect(roi)
              if whitecount>maxwhitecount:
                 maxwhitecount=whitecount
                 c2=x
                 r2=y
                 h2=h
                 w2=w


      #print(c2,r2,w2,h2)
      #cv2.rectangle(framecopy, (c2, r2), (c2 + w2, r2+h2), (0,200,0))#green
      #cv2.putText(framecopy, "Detecting_P2", (c2, r2), fontType, 1, (0, 0, 255), 1)


      bbox2= [c2,r2,w2,h2]#x座標、y座標、横幅、縦幅をbboxに格納
      bbox2= [float(i) for i in bbox2]
      bbox2=tuple(bbox2)
      tracker2.clear();
      tracker2= cv2.TrackerKCF_create()
      ok2 = tracker2.init(framecopy,bbox2)
      #print(bbox2)
      #print(ok2)
      ok2, bbox2 = tracker2.update(frame)#frameを入力、成功とbbのざひょうを返す





          #cv2.rectangle(frame, (x, y), (x + w2, y+h2), (0,0,200), 3)
          #ok2 = tracker2.init(frame, bbox2)
          #ok2, bbox2 = tracker2.update(frame)


    #####################フレーム差分によるボール検出##############################



    ############################フレームの更新#######################################
    #frame1 = frame2
    frame2 = frame3
    frame3 = cv2.cvtColor(video.read()[1], cv2.COLOR_RGB2GRAY)
    #cv2.setMouseCallback("framecopy", mouse_event,recentball)
    # フレームを画面表示
    cv2.imshow("framecopy", framecopy)


    #読み込んだフレームを書き込み
    out.write(framecopy)
    # ESCを押したら中止
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

video.release()
out.release()
cv2.destroyAllWindows()

'''
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
for i in range(len(ball_pos)):
    plt.plot(ball_pos[i][0], ball_pos[i][1],'o')
plt.legend()
# グラフの描画実行
plt.show()
'''
