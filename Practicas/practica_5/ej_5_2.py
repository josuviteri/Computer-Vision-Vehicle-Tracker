import cv2 as cv


# Cargar la imagen y convertirla a escala de grises
imagen = cv.imread("media/glass-caster-RGB.tif")


grises = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)



orb = cv.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=15)


kp_orb, des_orb = orb.detectAndCompute(grises, None)



# Dibujar puntos clave ORB
imagen_orb = cv.drawKeypoints(imagen, kp_orb, None, color=(0, 255, 0), 
                            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print("\n2. DETECTOR FAST + DESCRIPTOR BRIEF")

# Detector FAST
fast = cv.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)


# Descriptor BRIEF

try:
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
except Exception:
    # OpenCV may be built without xfeatures2d (contrib); fall back to BRISK
    brief = cv.BRISK_create()
    print("cv.xfeatures2d not available")
    

print("\n3. ALGORITMO SIFT")
sift = cv.SIFT_create(nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)




