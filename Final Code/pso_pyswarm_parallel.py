from functools import partial
import numpy as np
from copy import deepcopy
import cv2
from utils_mudSlap import *
from utils_naturalPerturbations import addFog, addRain

'''
HELPER Functions
'''
def makeObservations(imgList, mode='multi', farScale = 0.5, obliquePercentage = 10, obliqueDirection = 'left'):
    """
    Changes all the images in the provided list according to the given parameters.
        imgList: [ndarray]
            List of all images to be altered; All images must be of same dimensions
        mode: string ('single' or 'multi')
            If 'multi': ignores other parameters and generate 28 new observations for
                        each image. 7 different oblique patterns each at 4 different
                        distances (including original image too)
            If 'single': create 1 observation for each image in the list according to
                         the specified parameters
        farScale: scalar (float) (0-1]
            Parameter defining the closeness of the new image (1=same, ~0=farthest)
        obliquePercentage: scalar (int) (0-45)
            Defines how much the image is to be tilted
        obliqueDirection: string ('left' or 'right')
            Specifies the direction of tilt
    """
    if (len(imgList.shape)==3): #Only one image is provided
        numImgs = 1
        imgRows = imgList.shape[0]
        imgCols = imgList.shape[1]
        nChannels = imgList.shape[2]
        imgList = [imgList]
    else:
        numImgs = imgList.shape[0]
        imgRows = imgList.shape[1]
        imgCols = imgList.shape[2]
        nChannels = imgList.shape[3]
    if mode=='multi':
        obliquePercentages = [20, 30]
        farScales = [1, 0.25]
        increaseFactor = ((2*len(obliquePercentages))+1)*len(farScales)
        newImgList = np.zeros((numImgs*increaseFactor, imgRows, imgCols, nChannels))
        i = 0
        for img in imgList:
            for fS in farScales:
                img1 = cv2.resize(img, (int(imgRows*fS), int(imgCols*fS)))
                img1 = cv2.resize(img1, (imgRows, imgCols))
                img1 = cv2.GaussianBlur(img1, (5,5), 0)
                newImgList[i, ...] = img1
                i+=1
            for oP in obliquePercentages:
                for fS in farScales:
                    #For Left Oblique
                    XobliquePixels = oP*imgRows/100
                    YobliquePixels = 0.75*XobliquePixels
                    src = np.array([
                            [0, 0],
                            [imgRows - 1, 0],
                            [imgRows - 1, imgCols - 1],
                            [0, imgCols - 1]
                            ], dtype = "float32")
                    dst = np.array([
                            [XobliquePixels, YobliquePixels],
                            [imgRows - XobliquePixels - 1, 0],
                            [imgRows - XobliquePixels - 1, imgCols - 1],
                            [XobliquePixels, imgCols - YobliquePixels - 1]
                            ], dtype = "float32")
                    # compute the perspective transform matrix and then apply it
                    M = cv2.getPerspectiveTransform(src, dst)
                    warped = cv2.warpPerspective(img, M, (imgRows, imgCols))
                    #warped = warped[:, int(XobliquePixels):int(imgRows - XobliquePixels - 1)]
                    warped = warped[int(oP*YobliquePixels/100):int(imgCols - (oP*YobliquePixels/100) - 1),
                                int(XobliquePixels):int(imgRows - XobliquePixels - 1)]
                    warped = cv2.resize(warped, (imgRows, imgCols))
                    img1 = cv2.resize(warped, (int(imgRows*fS), int(imgCols*fS)))
                    img1 = cv2.resize(img1, (imgRows, imgCols))
                    img1 = cv2.GaussianBlur(img1, (5,5), 0)
                    newImgList[i, ...] = img1
                    i+=1
                    #For Right Oblique
                    XobliquePixels = oP*imgRows/100
                    YobliquePixels = 0.75*XobliquePixels
                    src = np.array([
                            [0, 0],
                            [imgRows - 1, 0],
                            [imgRows - 1, imgCols - 1],
                            [0, imgCols - 1]
                            ], dtype = "float32")
                    dst = np.array([
                            [XobliquePixels, 0],
                            [imgRows - XobliquePixels - 1, YobliquePixels],
                            [imgRows - XobliquePixels - 1, imgCols - YobliquePixels - 1],
                            [XobliquePixels, imgCols - 1]
                            ], dtype = "float32")
                    # compute the perspective transform matrix and then apply it
                    M = cv2.getPerspectiveTransform(src, dst)
                    warped = cv2.warpPerspective(img, M, (imgRows, imgCols))
                    warped = warped[int(oP*YobliquePixels/100):int(imgCols - (oP*YobliquePixels/100) - 1),
                                int(XobliquePixels):int(imgRows - XobliquePixels - 1)]
                    warped = cv2.resize(warped, (imgRows, imgCols))
                    img1 = cv2.resize(warped, (int(imgRows*fS), int(imgCols*fS)))
                    img1 = cv2.resize(img1, (imgRows, imgCols))
                    img1 = cv2.GaussianBlur(img1, (5,5), 0)
                    newImgList[i, ...] = img1
                    i+=1
    else:
        newImgList = np.zeros((numImgs, imgRows, imgCols, nChannels))
        increaseFactor = 1
        i = 0
        for img in imgList:
            XobliquePixels = obliquePercentage*imgRows/100
            YobliquePixels = 0.75*XobliquePixels
            if obliqueDirection=='left' or obliqueDirection == 'Left':
                src = np.array([
                    [0, 0],
                    [imgRows - 1, 0],
                    [imgRows - 1, imgCols - 1],
                    [0, imgCols - 1]
                    ], dtype = "float32")
                dst = np.array([
                    [XobliquePixels, YobliquePixels],
                    [imgRows - XobliquePixels - 1, 0],
                    [imgRows - XobliquePixels - 1, imgCols - 1],
                    [XobliquePixels, imgCols - YobliquePixels - 1]
                    ], dtype = "float32")
            else:
                src = np.array([
                    [0, 0],
                    [imgRows - 1, 0],
                    [imgRows - 1, imgCols - 1],
                    [0, imgCols - 1]
                    ], dtype = "float32")
                dst = np.array([
                    [XobliquePixels, 0],
                    [imgRows - XobliquePixels - 1, YobliquePixels],
                    [imgRows - XobliquePixels - 1, imgCols - YobliquePixels - 1],
                    [XobliquePixels, imgCols - 1]
                    ], dtype = "float32")
            # compute the perspective transform matrix and then apply it
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img, M, (imgRows, imgCols))
            warped = warped[int(obliquePercentage*YobliquePixels/100):int(imgCols - (obliquePercentage*YobliquePixels/100) - 1),
                        int(XobliquePixels):int(imgRows - XobliquePixels - 1)]
            warped = cv2.resize(warped, (imgRows, imgCols))
            img1 = cv2.resize(warped, (int(imgRows*farScale), int(imgCols*farScale)))
            img1 = cv2.resize(img1, (imgRows, imgCols))
            img1 = cv2.GaussianBlur(img1, (5,5), 0)
            newImgList[i, ...] = img1
            i+=1
    return newImgList, increaseFactor

def alterImages(imageList, alterFeatures = None, withRain = True):
    """
    Given a feature vector of alterations (as described below), alters a set of
    images accordingly

    Arguments:
    -----------
        imageList: np.ndarray (numImages, Width, Height, numChannels)
            List of images on which alterations are required to be added
        alterFeatures: List [<mudSplat1>, <mudSplat2>, <mudSplat3>, <rain>, <fog>]
            List of features to explain modifications in the order as defined above.
            Further explanations of each component as follows:
                <mudSplat1> : [mudSpaltObject]
                <mudSplat2> : [mudSpaltObject]
                <mudSplat3> : [mudSpaltObject]
                <rain>      : [randomSeed, rainy]
                <fog>       : [fogIntensity, randomSeed]
        withRain: boolean
            If False, will ignore the rain features
    Returns:
    -----------
        outImgs: np.ndarray (numImages, Width, Height, numChannels)
            The list of transformed output images with 'alterFeatures' effects
    """
    if len(imageList.shape)==3:
        imageList = np.expand_dims(imageList, axis=0)
    numImgs = imageList.shape[0]
    W = imageList.shape[1]
    H = imageList.shape[2]
    nCh = imageList.shape[3]
    if alterFeatures is not None:
        mudObj1  = alterFeatures[0]
        mudObj2  = alterFeatures[1]
        mudObj3  = alterFeatures[2]
        rainSeed = alterFeatures[3]
        rainExt  = alterFeatures[4]
        fogInten = alterFeatures[5]
        fogSeed  = alterFeatures[6]

        if mudObj1.scale>0 and mudObj2.scale>0 and mudObj3.scale>0:
            allSplatImg = combineSplats([mudObj1]+[mudObj2]+[mudObj3], W, H).astype('uint8')
            allSplatImg[:,:,:-1][allSplatImg[:,:,:-1]==0] = 255
            splatImgs = np.zeros(imageList.shape)
            for i, image in enumerate(imageList):
                splatImgs[i, ...] = addMudSplat(image, allSplatImg)
        elif mudObj1.scale>0 and mudObj2.scale>0:
            allSplatImg = combineSplats([mudObj1]+[mudObj2], W, H).astype('uint8')
            allSplatImg[:,:,:-1][allSplatImg[:,:,:-1]==0] = 255
            splatImgs = np.zeros(imageList.shape)
            for i, image in enumerate(imageList):
                splatImgs[i, ...] = addMudSplat(image, allSplatImg)
        elif mudObj1.scale>0:
            allSplatImg = combineSplats([mudObj1], W, H).astype('uint8')
            allSplatImg[:,:,:-1][allSplatImg[:,:,:-1]==0] = 255
            splatImgs = np.zeros(imageList.shape)
            for i, image in enumerate(imageList):
                splatImgs[i, ...] = addMudSplat(image, allSplatImg)
        else:
            splatImgs = deepcopy(imageList).astype("float64")
        if np.ceil(rainExt)>0 and withRain and np.ceil(fogInten)>0:
            outImgs = addRain(addFog(splatImgs, fogInten, int(fogSeed)), int(rainSeed))
        elif np.ceil(rainExt)>0 and withRain:
            outImgs = addRain(splatImgs, int(rainSeed))
        elif np.ceil(fogInten)>0:
            outImgs = addFog(splatImgs, fogInten, int(fogSeed))
        else:
            outImgs = splatImgs
    else:
        outImgs = imageList
    return outImgs

def predictModel_Nparticles(originalImages, originalClass, targetClass,
                         model, alterFeatures = None):
    """
    Adds a mud splat on the original image according to the specified feature
    vector and then predict its class using the trained model.

    Arguments:
    -----------
        originalImages: np.ndarray (numImages, Width, Height, numChannels)
            List of original images (without any perturbations)
        originalClass: scalar (int) {0,1,2}
            Original Class label specified as follows:
                turnLeft: 0
                turnRight: 1
                goStraight: 2
        targetClass: scalar (int) {0,1,2}
            Target Class (aiming to misclassify) label specified as follows:
                turnLeft: 0
                turnRight: 1
                goStraight: 2
        trainedModelWeightsPath: Keras Model Object
            Previously Trained Model Object
        alterFeatures: List [[<mudSplat1>, <mudSplat2>, <mudSplat3>, <rain>, <fog>]]
            List of list of features to explain modifications in the order as defined above.
            **Must be either of length 1 or equal to the number of originalImages or None.
            Further explanations of each component as follows:
                <mudSplat1> : [mudSpaltObject]
                <mudSplat2> : [mudSpaltObject]
                <mudSplat3> : [mudSpaltObject]
                <rain>      : [randomSeed]
                <fog>       : [fogIntensity, randomSeed]
    Returns:
    -----------
        predScore: Classify predictions w.r.t. original and target classes and
            give score according to the following scheme:
                2   : if predicted class == original class (correct prediction)
                0   : if predicted class == target class (incorrect prediction -
                                                          target matched)
                1   : if predicted class == any other class (incorrect prediction -
                                                             target mismatched)
            The scores for all images are then added to produce final score
        predOutput: list of predictions
            The transformed output image with mud-splat on it
    """
    print("Starting to alter images...")
    if len(originalImages.shape)==3:
        originalImages = np.expand_dims(originalImages, axis=0)
    numImgs = originalImages.shape[0]
    W = originalImages.shape[1]
    H = originalImages.shape[2]
    nCh = originalImages.shape[3]

    if alterFeatures is None:
        raise Exception('alterFeatures not provided for any particle')
    else:
        numFinImages = numImgs*len(alterFeatures)
        finImages = np.zeros((numFinImages, W, H, nCh))
        for n, feature in enumerate(alterFeatures):
            finImages[n*numImgs:(n+1)*numImgs, ...] = alterImages(originalImages, feature, False)
        print("Alteration process is success. Proceeding to making observations.")
        finImages, increaseFactor = makeObservations(finImages)
        numImgs = numImgs*increaseFactor
        for n, feature in enumerate(alterFeatures):
            if np.ceil(feature[4])>0:
                finImages[n*numImgs:(n+1)*numImgs, ...] = addRain(finImages[n*numImgs:(n+1)*numImgs, ...].astype("float64"), int(feature[3]))
        finImages = np.uint8(finImages)
        print("Made observations successfully. Proceeding to model predictions.")
        predOutput = model.predict(finImages)

    predScore = np.zeros(len(alterFeatures))
    for n in range(len(alterFeatures)):
        for outputs in predOutput[n*numImgs:(n+1)*numImgs]:
            if np.any(np.argmax(outputs) == originalClass):
                predScore[n] += 2
            elif np.any(np.argmax(outputs) == targetClass):
                predScore[n] += 0
            else:
                predScore[n] += 1
    print("Predictions made and scores calculated. Returning handle back to PSO.")
    print("For reference, scores are:")
    print(predScore)
    return predScore, predOutput
'''
def  predictModelMudSplat_Nparticles(originalImages, originalClass, targetClass,
                         model, mudSplatObjects = None):
    """
    Adds a mud splat on the original image according to the specified feature
    vector and then predict its class using the trained model.

    Arguments:
    -----------
        originalImages: np.ndarray
            Original Image (without any perturbations)
        originalClass: scalar (int) {0,1,2}
            Original Class label specified as follows:
                turnLeft: 0
                turnRight: 1
                goStraight: 2
        targetClass: scalar (int) {0,1,2}
            Target Class (aiming to misclassify) label specified as follows:
                turnLeft: 0
                turnRight: 1
                goStraight: 2
        trainedModelWeightsPath: Keras Model Object
            Previously Trained Model Object
        mudSplatObjects: Array of mud-splat objects
            Explains the features of all mud-splats.
            (If None: Predict the class and correctness of original image.)
    Returns:
    -----------
        predScore: Classify predictions w.r.t. original and target classes and
            give score according to the following scheme:
                2   : if predicted class == original class (correct prediction)
                0   : if predicted class == target class (incorrect prediction -
                                                          target matched)
                1   : if predicted class == any other class (incorrect prediction -
                                                             target mismatched)
            The scores for all images are then added to produce final score
        predOutput: list of predictions
            The transformed output image with mud-splat on it
    """
    print("Here")
    newImages = deepcopy(originalImages)
    if len(newImages.shape) == 3:   # Only 1 image is provided
        newImages = np.expand_dims(newImages, axis=0)
    if mudSplatObjects is not None:
        numSplattedImages = newImages.shape[0]*len(mudSplatObjects)
        splattedImages = np.zeros(np.append(numSplattedImages,newImages.shape[1:]))
        print(len(mudSplatObjects))
        print(len(newImages))
        for n, mudSplatObj in enumerate(mudSplatObjects):
            mudSplat = cv2.imread(mudSplatObj.imgPath, cv2.IMREAD_UNCHANGED)
            for i, img in enumerate(newImages):
                splattedImages[n*len(newImages)+i,...] = addMudSplat(img, mudSplat, mudSplatObj.xOffset,
                                                                    mudSplatObj.yOffset, mudSplatObj.scale,
                                                                    mudSplatObj.rotate)
        predOutput = model.predict(splattedImages)
    else:
        predOutput = model.predict(newImages)
    print("Here Again!")
    predScore = np.zeros(len(mudSplatObjects))
    for n in range(len(mudSplatObjects)):
        for outputs in predOutput[n*len(newImages):(n+1)*len(newImages)]:
            if np.any(np.argmax(outputs) == originalClass):
                predScore[n] += 2
            elif np.any(np.argmax(outputs) == targetClass):
                predScore[n] += 0
            else:
                predScore[n] += 1
    print("Here Too")
    print(predScore)
    return predScore, predOutput
'''
def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)

def _is_feasible_wrapper(func, x):
    return np.all(func(x)>=0)

def _cons_none_wrapper(x):
    return np.array([0])

def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])

def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))

def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
        minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
        particle_output=False, abs_min=None):
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)

    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified,
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.
    abs_min : int
        If 'abs_min' is given, iterations will terminate if the value of PSO
        objective function falls below or equal to this number

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p

    """

    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>=lb), 'All upper-bound values must be greater than or equal to lower-bound values'

    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args, kwargs)

    # Check for constraint function(s) #########################################
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)

    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S)*np.inf  # best particle function values
    g = []  # best swarm position
    fg = np.inf  # best swarm position starting value

    # Initialize the particle's position
    x = lb + x*(ub - lb)
    #x[-1,:] = [25, 50, 40, 220, 55, 20, 30, 150, 60, 40, 20, 90, 0, 0, 0, 0]
    x[:,2] = ub[2]
    x[:,6] = ub[6]
    x[:,10]= ub[10]
    x[-1,14]= ub[14]

    # Calculate objective and constraints for each particle
    if processes > 1:
        fx = np.array(mp_pool.map(obj, x))
        fs = np.array(mp_pool.map(is_feasible, x))
    else:
        imgData, oriClass, tarClass, mudImgPath, model = args
        allFeatures = []
        for i in range(S):
            mudSplatObject1 = mudSplat(mudImgPath, int(x[i,0]), int(x[i,1]), x[i,2], x[i,3])
            mudSplatObject2 = mudSplat(mudImgPath, int(x[i,4]), int(x[i,5]), x[i,6], x[i,7])
            mudSplatObject3 = mudSplat(mudImgPath, int(x[i,8]), int(x[i,9]), x[i,10], x[i,11])
            rainFeatures    = [int(x[i,12]), np.ceil(x[i,13])]
            fogFeatures     = [x[i,14], int(x[i,15])]
            allFeatures.append([mudSplatObject1] + [mudSplatObject2] + [mudSplatObject3] + rainFeatures + fogFeatures)
        fx = predictModel_Nparticles(imgData, oriClass, tarClass, model, allFeatures)[0]

        for i in range(S):
            #fx[i] = obj(x[i, :])
            fs[i] = is_feasible(x[i, :])

    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()

    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)

    # Iterate until termination criterion met ##################################
    fg_history = []
    fg_history.append(fg)
    it = 1
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        # Update the particles velocities
        v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)

        # Update the particles' positions
        x = x + v

        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku

        # Update objectives and constraints
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            imgData, oriClass, tarClass, mudImgPath, model = args
            allFeatures = []
            for i in range(S):
                X = deepcopy(x[i,:])
                mudSplatObject1 = mudSplat(mudImgPath, int(X[0]), int(X[1]), X[2], X[3])
                mudSplatObject2 = mudSplat(mudImgPath, int(X[4]), int(X[5]), X[6], X[7])
                mudSplatObject3 = mudSplat(mudImgPath, int(X[8]), int(X[9]), X[10], X[11])
                rainFeatures    = [int(X[12]), np.ceil(X[13])]
                fogFeatures     = [X[14], int(X[15])]
                allFeatures.append([mudSplatObject1] + [mudSplatObject2] + [mudSplatObject3] + rainFeatures + fogFeatures)
            fx = predictModel_Nparticles(imgData, oriClass, tarClass, model, allFeatures)[0]

            for i in range(S):
                #fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:
                print('New best for swarm at iteration {:}: {:} {:}'\
                    .format(it, p[i_min, :], fp[i_min]))

            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min)**2))

            if np.abs(fg - fp[i_min]) <= minfunc:
                print('Stopping search: Swarm best objective change less than {:}'\
                    .format(minfunc))
                if particle_output:
                    return p_min, fp[i_min], fg_history, p, fp
                else:
                    return p_min, fp[i_min], fg_history
            elif stepsize <= minstep:
                print('Stopping search: Swarm best position change less than {:}'\
                    .format(minstep))
                if particle_output:
                    return p_min, fp[i_min], fg_history, p, fp
                else:
                    return p_min, fp[i_min], fg_history
            else:
                g = p_min.copy()
                fg = fp[i_min]

        if fg<=abs_min:
            print('Stopping search: Swarm best position change less than or equal to given absolute minima{:}'\
                .format(abs_min))
            if debug:
                print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            if particle_output:
                return g, fg, fg_history, p, fp
            else:
                return g, fg, fg_history
        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        fg_history.append(fg)
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))

    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    if particle_output:
        return g, fg, fg_history, p, fp
    else:
        return g, fg, fg_history
