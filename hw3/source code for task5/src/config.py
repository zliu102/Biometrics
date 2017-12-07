## parameters about dataset
N_CLASSES = 15
N_IMAGES_PER_CLASS = 11
N_TRAINING_SAMPLES_PER_CLASS = 6 # MUST < N_IMAGES_PER_CLASS

## parameters about preprocessing:
CLIP_RATIO = 0.8
PP_GAMMA = 0.4
PP_SIGMA1 = 1
PP_SIGMA2 = 2
PP_ALPHA = 0.1
PP_TAU = 10

## parameters about eigenfaces
EF_N_FEATURE_DIMS_PRESERVED = 32 # Influnce results of Laplacian and Fisher faces

## parameters about Laplacianfaces
LF_K_NEAREST_NEIGHBORS = 7
LF_N_DIMS_PCA = 84
LF_N_FEATURE_DIMS_PRESERVED = 32

## parameters about Fisherfaces
FF_N_DIMS_PCA = 71 # MUST <= N_CLASSES * (N_TRAINING_SAMPLES_PER_CLASS - 1)

# Note that scatter matrix after PCA projection can still be singular,
# which will cause the failure of solving procedure. When it happens,
# try reducing LF_N_DIMS_PCA for Laplacianfaces or reducing 
# FF_N_DIMS_PCA for Fisherfaces.