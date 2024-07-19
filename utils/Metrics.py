import numpy as np
import math
import scipy.ndimage
from scipy.signal import convolve2d
from .Qabf import get_Qabf
from .Nabf import get_Nabf
from .ssim import ssim, ms_ssim

def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]) #分别对应通道 R G B

def cross_covariance(x, y, mu_x, mu_y):
    return 1 / (x.size - 1) * np.sum((x - mu_x) * (y - mu_y))

#------------MSE------------#
def MSE_function(A, B, F):
    if len(A.shape) == 3:
        A = rgb2gray(A)
    if len(B.shape) == 3:
        B = rgb2gray(B)
    if len(F.shape) == 3:
        F = rgb2gray(F)
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A)**2))/(m*n)
    MSE_BF = np.sum(np.sum((F - B)**2))/(m*n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    return MSE

#------------SSIM------------#
def SSIM_function(x, y):
    if len(x.shape) == 3:
        x = rgb2gray(x)
    if len(y.shape) == 3:
        y = rgb2gray(y)

    L = np.max(np.array([x, y])) - np.min(np.array([x, y]))
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sig_x = np.std(x)
    sig_y = np.std(y)
    sig_xy = cross_covariance(x, y, mu_x, mu_y)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2
    return (2 * mu_x * mu_y + C1) * (2 * sig_x * sig_y + C2) * (sig_xy + C3) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2) * (sig_x * sig_y + C3))

#------------PSNR------------#
def PSNR_function(img1, img2):
    # img1 = img1.astype(np.float64)
    # img2 = img2.cpu().numpy()
    # # img2 = np.transpose(img2, (1, 2, 0))
    # img2 = img2.astype(np.uint8)
    if len(img1.shape) == 3:
        img1 = rgb2gray(img1)
    if len(img2.shape) == 3:
        img2 = rgb2gray(img2)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


#------------EN------------#
def EN_function(image_array):
    # 计算图像的直方图
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 255))
    # 将直方图归一化
    histogram = histogram / float(np.sum(histogram))
    # 计算熵
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy


#------------SF------------#
def SF_function(image):
    if len(image.shape) == 3:
        image = rgb2gray(image)
    image_array = np.array(image)
    RF = np.diff(image_array, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(image_array, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF


#------------SD------------#
def SD_function(image_array):
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)
    m, n = image_array.shape
    u = np.mean(image_array)
    SD = np.sqrt(np.sum(np.sum((image_array - u) ** 2)) / (m * n))
    return SD

#------------VIFF------------#
def VIFF_function(image_F, image_A, image_B):
    refA=image_A
    refB=image_B
    dist=image_F

    sigma_nsq = 2
    eps = 1e-10
    numA = 0.0
    denA = 0.0
    numB = 0.0
    denB = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        # Create a Gaussian kernel as MATLAB's
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            refA = refA[::2, ::2]
            refB = refB[::2, ::2]
            dist = dist[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq_A = mu1A * mu1A
        mu1_sq_B = mu1B * mu1B
        mu2_sq = mu2 * mu2
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2
        sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
        sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
        sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

        sigma1A_sq[sigma1A_sq < 0] = 0
        sigma1B_sq[sigma1B_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        gA = sigma12_A / (sigma1A_sq + eps)
        gB = sigma12_B / (sigma1B_sq + eps)
        sv_sq_A = sigma2_sq - gA * sigma12_A
        sv_sq_B = sigma2_sq - gB * sigma12_B

        gA[sigma1A_sq < eps] = 0
        gB[sigma1B_sq < eps] = 0
        sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
        sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
        sigma1A_sq[sigma1A_sq < eps] = 0
        sigma1B_sq[sigma1B_sq < eps] = 0

        gA[sigma2_sq < eps] = 0
        gB[sigma2_sq < eps] = 0
        sv_sq_A[sigma2_sq < eps] = 0
        sv_sq_B[sigma2_sq < eps] = 0

        sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
        sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
        gA[gA < 0] = 0
        gB[gB < 0] = 0
        sv_sq_A[sv_sq_A <= eps] = eps
        sv_sq_B[sv_sq_B <= eps] = eps

        numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
        numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
        denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
        denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

    vifpA = numA / denA
    vifpB =numB / denB

    if np.isnan(vifpA):
        vifpA=1
    if np.isnan(vifpB):
        vifpB = 1
    return vifpA+vifpB


#------------CC------------#
def CC_function(A,B,F):
    if len(A.shape) == 3:
        A = rgb2gray(A)
    if len(B.shape) == 3:
        B = rgb2gray(B)
    if len(F.shape) == 3:
        F = rgb2gray(F)
    rAF = np.sum((A - np.mean(A)) * (F - np.mean(F))) / np.sqrt(np.sum((A - np.mean(A)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    rBF = np.sum((B - np.mean(B)) * (F - np.mean(F))) / np.sqrt(np.sum((B - np.mean(B)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    CC = np.mean([rAF, rBF])
    return CC

#------------SCD------------#
def corr2(a, b):
    if len(a.shape) == 3:
        a = rgb2gray(a)
    if len(b.shape) == 3:
        b = rgb2gray(b)

    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r

def SCD_function(A, B, F):
    if len(A.shape) == 3:
        A = rgb2gray(A)
    if len(B.shape) == 3:
        B = rgb2gray(B)
    if len(F.shape) == 3:
        F = rgb2gray(F)
    r = corr2(F - B, A) + corr2(F - A, B)
    return r

#------------Qabf------------#
def Qabf_function(A, B, F):
    if len(A.shape) == 3:
        A = rgb2gray(A)
    if len(B.shape) == 3:
        B = rgb2gray(B)
    if len(F.shape) == 3:
        F = rgb2gray(F)
    return get_Qabf(A, B, F)

#------------Nabf------------#
# def Nabf_function(A, B, F):
#     if len(A.shape) == 3:
#         A = rgb2gray(A)
#     if len(B.shape) == 3:
#         B = rgb2gray(B)
#     if len(F.shape) == 3:
#         F = rgb2gray(F)
#     return Nabf_function(A, B, F)

#------------MI------------#
def Hab(im1, im2, gray_level):
    if len(im1.shape) == 3:
        im1 = rgb2gray(im1)
    if len(im2.shape) == 3:
        im2 = rgb2gray(im2)

    hang, lie = im1.shape
    count = hang * lie
    N = gray_level
    h = np.zeros((N, N))
    for i in range(hang):
        for j in range(lie):
            h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1
    h = h / np.sum(h)
    im1_marg = np.sum(h, axis=0)
    im2_marg = np.sum(h, axis=1)
    H_x = 0
    H_y = 0
    for i in range(N):
        if (im1_marg[i] != 0):
            H_x = H_x + im1_marg[i] * math.log2(im1_marg[i])
    for i in range(N):
        if (im2_marg[i] != 0):
            H_x = H_x + im2_marg[i] * math.log2(im2_marg[i])
    H_xy = 0
    for i in range(N):
        for j in range(N):
            if (h[i, j] != 0):
                H_xy = H_xy + h[i, j] * math.log2(h[i, j])
    MI = H_xy - H_x - H_y
    return MI

def MI_function(A, B, F, gray_level=256):
	MIA = Hab(A, F, gray_level)
	MIB = Hab(B, F, gray_level)
	MI_results = MIA + MIB
	return MI_results


#------------AG------------#
def AG_function(image):
    if len(image.shape) == 3:
        image = rgb2gray(image)
    width = image.shape[1]
    width = width - 1
    height = image.shape[0]
    height = height - 1
	# tmp = 0.0
    [grady, gradx] = np.gradient(image)
    s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
    AG = np.sum(np.sum(s)) / (width * height)
    return AG

#------------SSIM------------#
def SSIM_function(A, B, F):
    ssim_A = ssim(A, F)
    ssim_B = ssim(B, F)
    SSIM = 1 * ssim_A + 1 * ssim_B
    return SSIM.item()

#------------MS-SSIM------------#
def MS_SSIM_function(A, B, F):
    ssim_A = ms_ssim(A, F)
    ssim_B = ms_ssim(B, F)
    MS_SSIM = 1 * ssim_A + 1 * ssim_B
    return MS_SSIM.item()

#------------Nabf------------#
def Nabf_function(A, B, F):
    Nabf = get_Nabf(A, B, F)
    return Nabf
