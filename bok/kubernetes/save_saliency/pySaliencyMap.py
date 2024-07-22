import cv2
import numpy as np
import pySaliencyMapDefs

class pySaliencyMap:
    # 초기화
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.prev_frame = None
        self.SM = None
        self.GaborKernel0   = np.array(pySaliencyMapDefs.GaborKernel_0)
        self.GaborKernel45  = np.array(pySaliencyMapDefs.GaborKernel_45)
        self.GaborKernel90  = np.array(pySaliencyMapDefs.GaborKernel_90)
        self.GaborKernel135 = np.array(pySaliencyMapDefs.GaborKernel_135)

    # 색상 채널 추출
    def SMExtractRGBI(self, inputImage):
        # 배열 요소의 스케일 변환
        src = np.float32(inputImage) * 1./255
        # 분할
        (B, G, R) = cv2.split(src)
        # 강도 이미지 추출
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # 반환
        return R, G, B, I

    # 특징 맵
    ## 가우시안 피라미드 생성
    def FMCreateGaussianPyr(self, src):
        dst = list()
        dst.append(src)
        for i in range(1,9):
            nowdst = cv2.pyrDown(dst[i-1])
            dst.append(nowdst)
        return dst

    ## 중심-주변 차이 계산
    def FMCenterSurroundDiff(self, GaussianMaps):
        dst = list()
        for s in range(2,5):
            now_size = GaussianMaps[s].shape
            now_size = (now_size[1], now_size[0])  ## (width, height)
            tmp = cv2.resize(GaussianMaps[s+3], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
            tmp = cv2.resize(GaussianMaps[s+4], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
        return dst

    ## 가우시안 피라미드 생성 및 중심-주변 차이 계산
    def FMGaussianPyrCSD(self, src):
        GaussianMaps = self.FMCreateGaussianPyr(src)
        dst = self.FMCenterSurroundDiff(GaussianMaps)
        return dst

    ## 강도 특징 맵
    def IFMGetFM(self, I):
        return self.FMGaussianPyrCSD(I)

    ## 색상 특징 맵
    def CFMGetFM(self, R, G, B):
        # max(R,G,B)
        tmp1 = cv2.max(R, G)
        RGBMax = cv2.max(B, tmp1)
        RGBMax[RGBMax <= 0] = 0.0001    # 0으로 나누는 것을 방지
        # min(R,G)
        RGMin = cv2.min(R, G)
        # RG = (R-G)/max(R,G,B)
        RG = (R - G) / RGBMax
        # BY = (B-min(R,G)/max(R,G,B)
        BY = (B - RGMin) / RGBMax
        # 음수 값은 0으로 클램프
        RG[RG < 0] = 0
        BY[BY < 0] = 0
        # 강도와 같은 방식으로 특징 맵 얻기
        RGFM = self.FMGaussianPyrCSD(RG)
        BYFM = self.FMGaussianPyrCSD(BY)
        # 반환
        return RGFM, BYFM

    ## 방향 특징 맵
    def OFMGetFM(self, src):
        # 가우시안 피라미드 생성
        GaussianI = self.FMCreateGaussianPyr(src)
        # 강도 이미지에 가보 필터 적용하여 방향 특징 추출
        GaborOutput0   = [ np.empty((1,1)), np.empty((1,1)) ]  # 더미 데이터: 어떤 종류의 np.array()라도 상관 없음
        GaborOutput45  = [ np.empty((1,1)), np.empty((1,1)) ]
        GaborOutput90  = [ np.empty((1,1)), np.empty((1,1)) ]
        GaborOutput135 = [ np.empty((1,1)), np.empty((1,1)) ]
        for j in range(2,9):
            GaborOutput0.append(   cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel0) )
            GaborOutput45.append(  cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel45) )
            GaborOutput90.append(  cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel90) )
            GaborOutput135.append( cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel135) )
        # 각 방향에 대해 중심-주변 차이 계산
        CSD0   = self.FMCenterSurroundDiff(GaborOutput0)
        CSD45  = self.FMCenterSurroundDiff(GaborOutput45)
        CSD90  = self.FMCenterSurroundDiff(GaborOutput90)
        CSD135 = self.FMCenterSurroundDiff(GaborOutput135)
        # 연결
        dst = list(CSD0)
        dst.extend(CSD45)
        dst.extend(CSD90)
        dst.extend(CSD135)
        # 반환
        return dst

    ## 움직임 특징 맵
    def MFMGetFM(self, src):
        # 스케일 변환
        I8U = np.uint8(255 * src)
        cv2.waitKey(10)
        # 옵티컬 플로우 계산
        if self.prev_frame is not None:
            farne_pyr_scale= pySaliencyMapDefs.farne_pyr_scale
            farne_levels = pySaliencyMapDefs.farne_levels
            farne_winsize = pySaliencyMapDefs.farne_winsize
            farne_iterations = pySaliencyMapDefs.farne_iterations
            farne_poly_n = pySaliencyMapDefs.farne_poly_n
            farne_poly_sigma = pySaliencyMapDefs.farne_poly_sigma
            farne_flags = pySaliencyMapDefs.farne_flags
            flow = cv2.calcOpticalFlowFarneback(
                prev = self.prev_frame,
                next = I8U,
                pyr_scale = farne_pyr_scale,
                levels = farne_levels,
                winsize = farne_winsize,
                iterations = farne_iterations,
                poly_n = farne_poly_n,
                poly_sigma = farne_poly_sigma,
                flags = farne_flags,
                flow = None
            )
            flowx = flow[...,0]
            flowy = flow[...,1]
        else:
            flowx = np.zeros(I8U.shape)
            flowy = np.zeros(I8U.shape)
        # 가우시안 피라미드 생성
        dst_x = self.FMGaussianPyrCSD(flowx)
        dst_y = self.FMGaussianPyrCSD(flowy)
        # 현재 프레임 업데이트
        self.prev_frame = np.uint8(I8U)
        # 반환
        return dst_x, dst_y

    # 주목성 맵
    ## 표준 범위 정규화
    def SMRangeNormalize(self, src):
        minn, maxx, dummy1, dummy2 = cv2.minMaxLoc(src)
        if maxx!=minn:
            dst = src/(maxx-minn) + minn/(minn-maxx)
        else:
            dst = src - minn
        return dst

    ## 지역 최대값의 평균 계산
    def SMAvgLocalMax(self, src):
        # 크기
        stepsize = pySaliencyMapDefs.default_step_local
        width = src.shape[1]
        height = src.shape[0]
        # 지역 최대값 찾기
        numlocal = 0
        lmaxmean = 0
        for y in range(0, height-stepsize, stepsize):
            for x in range(0, width-stepsize, stepsize):
                localimg = src[y:y+stepsize, x:x+stepsize]
                lmin, lmax, dummy1, dummy2 = cv2.minMaxLoc(localimg)
                lmaxmean += lmax
                numlocal += 1
        # 모든 지역 영역에 대한 평균 계산
        return lmaxmean / numlocal

    ## 주목성 맵 모델을 위한 정규화
    def SMNormalization(self, src):
        dst = self.SMRangeNormalize(src)
        lmaxmean = self.SMAvgLocalMax(dst)
        normcoeff = (1-lmaxmean)*(1-lmaxmean)
        return dst * normcoeff

    ## 특징 맵 정규화
    def normalizeFeatureMaps(self, FM):
        NFM = list()
        for i in range(0,6):
            normalizedImage = self.SMNormalization(FM[i])
            nownfm = cv2.resize(normalizedImage, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            NFM.append(nownfm)
        return NFM

    ## 강도 주목성 맵
    def ICMGetCM(self, IFM):
        NIFM = self.normalizeFeatureMaps(IFM)
        ICM = sum(NIFM)
        return ICM

    ## 색상 주목성 맵
    def CCMGetCM(self, CFM_RG, CFM_BY):
        # 각 색상 상대 쌍에 대해 주목성 맵 추출
        CCM_RG = self.ICMGetCM(CFM_RG)
        CCM_BY = self.ICMGetCM(CFM_BY)
        # 합치기
        CCM = CCM_RG + CCM_BY
        # 반환
        return CCM

    ## 방향 주목성 맵
    def OCMGetCM(self, OFM):
        OCM = np.zeros((self.height, self.width))
        for i in range (0,4):
            # 슬라이싱
            nowofm = OFM[i*6:(i+1)*6]  # 각도 = i*45
            # 각 각도에 대해 주목성 맵 추출
            NOFM = self.ICMGetCM(nowofm)
            # 정규화
            NOFM2 = self.SMNormalization(NOFM)
            # 누적
            OCM += NOFM2
        return OCM

    ## 움직임 주목성 맵
    def MCMGetCM(self, MFM_X, MFM_Y):
        return self.CCMGetCM(MFM_X, MFM_Y)

    # 핵심 함수
    def SMGetSM(self, src):
        # 정의
        size = src.shape
        width  = size[1]
        height = size[0]
        # 확인
#        if(width != self.width or height != self.height):
#            sys.exit("size mismatch")
        # 개별 색상 채널 추출
        R, G, B, I = self.SMExtractRGBI(src)
        # 특징 맵 추출
        IFM = self.IFMGetFM(I)
        CFM_RG, CFM_BY = self.CFMGetFM(R, G, B)
        OFM = self.OFMGetFM(I)
        MFM_X, MFM_Y = self.MFMGetFM(I)
        # 주목성 맵 추출
        ICM = self.ICMGetCM(IFM)
        CCM = self.CCMGetCM(CFM_RG, CFM_BY)
        OCM = self.OCMGetCM(OFM)
        MCM = self.MCMGetCM(MFM_X, MFM_Y)
        # 모든 주목성 맵을 더하여 주목성 맵 생성
        wi = pySaliencyMapDefs.weight_intensity
        wc = pySaliencyMapDefs.weight_color
        wo = pySaliencyMapDefs.weight_orientation
        wm = pySaliencyMapDefs.weight_motion
        SMMat = wi*ICM + wc*CCM + wo*OCM + wm*MCM
        # 정규화
        normalizedSM = self.SMRangeNormalize(SMMat)
        normalizedSM2 = normalizedSM.astype(np.float32)
        smoothedSM = cv2.bilateralFilter(normalizedSM2, 7, 3, 1.55)  # 양방향필터(이미지, 필터링에 사용될 픽셀의 거리, 색 공간에서 필터의 표준 편차, 좌표 공간에서 필터의 표준 편차)
        self.SM = cv2.resize(smoothedSM, (width,height), interpolation=cv2.INTER_NEAREST)
        # 반환
        return self.SM

    # 이진화된 주목성 맵 얻기
    def SMGetBinarizedSM(self, src):
        # 주목성 맵 얻기
        if self.SM is None:
            self.SM = self.SMGetSM(src)
        # 스케일 변환
        SM_I8U = np.uint8(255 * self.SM)
        # 이진화
        thresh, binarized_SM = cv2.threshold(SM_I8U, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binarized_SM

    # 주목성 영역 얻기
    def SMGetSalientRegion(self, src):
        # 이진화된 주목성 맵 얻기
        binarized_SM = self.SMGetBinarizedSM(src)
        # GrabCut
        img = src.copy()
        mask =  np.where((binarized_SM!=0), cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')  # FGD: 아마도 전경(3), # BGD: 아마도 배경(2)
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        rect = (0,0,1,1)  # 더미
        iterCount = 1
        cv2.grabCut(img, mask=mask, rect=rect, bgdModel=bgdmodel, fgdModel=fgdmodel, iterCount=iterCount, mode=cv2.GC_INIT_WITH_MASK)  # GC_INT_WITH_MASK: mask에 지정한 값을 기준으로 그랩컷 수행
        # 후처리
        mask_out = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask_out)  # mask_out 영역에서 공통으로 겹치는 부분 출력
        return output
