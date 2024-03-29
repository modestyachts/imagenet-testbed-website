from enum import Enum


class ModelTypes(Enum):
    STANDARD = ('Standard training', 'tab:blue')
    LP_ADV = ('Lp adversarially robust', 'tab:olive')
    ROBUST_INTV = ('Other robustness intervention', 'tab:brown')
    MORE_DATA = ('Trained with more data', 'tab:green')
    RANDOM_FEATURES = ('Random Features', 'tab:yellow')	
    LINEAR_PIXELS = ('Linear Classifier on Pixels', 'tab:orange')	
    RANDOM_FORESTS = ('Random Forests', 'tab:pink')
    NEAREST_NEIGHBORS = ('Nearest Neighbors', 'tab:purple')
    LOW_ACCURACY_CNN = ('Low Accuracy CNN', 'tab:cyan')

model_types_map = {
'BiT-M-R50x1-ILSVRC2012': ModelTypes.MORE_DATA,
'BiT-M-R50x3-ILSVRC2012': ModelTypes.MORE_DATA,
'BiT-M-R101x1-ILSVRC2012': ModelTypes.MORE_DATA,
'BiT-M-R101x3-ILSVRC2012': ModelTypes.MORE_DATA,
'BiT-M-R152x4-ILSVRC2012': ModelTypes.MORE_DATA,
'BiT-M-R50x1-nonfinetuned': ModelTypes.MORE_DATA,
'BiT-M-R50x3-nonfinetuned': ModelTypes.MORE_DATA,
'BiT-M-R101x1-nonfinetuned': ModelTypes.MORE_DATA,
'BiT-M-R101x3-nonfinetuned': ModelTypes.MORE_DATA,
'BiT-M-R152x4-nonfinetuned': ModelTypes.MORE_DATA,
'FixPNASNet': ModelTypes.STANDARD,
'FixResNeXt101_32x48d': ModelTypes.MORE_DATA,
'FixResNeXt101_32x48d_v2': ModelTypes.MORE_DATA,
'FixResNet50': ModelTypes.STANDARD,
'FixResNet50CutMix': ModelTypes.ROBUST_INTV,
'FixResNet50CutMix_v2': ModelTypes.ROBUST_INTV,
'FixResNet50_no_adaptation': ModelTypes.STANDARD,
'FixResNet50_v2': ModelTypes.STANDARD,
'alexnet': ModelTypes.STANDARD,
'alexnet_lpf2': ModelTypes.ROBUST_INTV,
'alexnet_lpf3': ModelTypes.ROBUST_INTV,
'alexnet_lpf5': ModelTypes.ROBUST_INTV,
'bninception': ModelTypes.STANDARD,
'bninception-imagenet21k': ModelTypes.MORE_DATA,
'cafferesnet101': ModelTypes.STANDARD,
'densenet121': ModelTypes.STANDARD,
'densenet121_lpf2': ModelTypes.ROBUST_INTV,
'densenet121_lpf3': ModelTypes.ROBUST_INTV,
'densenet121_lpf5': ModelTypes.ROBUST_INTV,
'densenet161': ModelTypes.STANDARD,
'densenet169': ModelTypes.STANDARD,
'densenet201': ModelTypes.STANDARD,
'dpn107': ModelTypes.MORE_DATA,
'dpn131': ModelTypes.STANDARD,
'dpn68': ModelTypes.STANDARD,
'dpn68b': ModelTypes.MORE_DATA,
'dpn92': ModelTypes.MORE_DATA,
'dpn98': ModelTypes.STANDARD,
'efficientnet-b0': ModelTypes.STANDARD,
'efficientnet-b0-advprop-autoaug': ModelTypes.ROBUST_INTV,
'efficientnet-b0-autoaug': ModelTypes.STANDARD,
'efficientnet-b1': ModelTypes.STANDARD,
'efficientnet-b1-advprop-autoaug': ModelTypes.ROBUST_INTV,
'efficientnet-b1-autoaug': ModelTypes.STANDARD,
'efficientnet-b2': ModelTypes.STANDARD,
'efficientnet-b2-advprop-autoaug': ModelTypes.ROBUST_INTV,
'efficientnet-b2-autoaug': ModelTypes.STANDARD,
'efficientnet-b3': ModelTypes.STANDARD,
'efficientnet-b3-advprop-autoaug': ModelTypes.ROBUST_INTV,
'efficientnet-b3-autoaug': ModelTypes.STANDARD,
'efficientnet-b4': ModelTypes.STANDARD,
'efficientnet-b4-advprop-autoaug': ModelTypes.ROBUST_INTV,
'efficientnet-b4-autoaug': ModelTypes.STANDARD,
'efficientnet-b5': ModelTypes.STANDARD,
'efficientnet-b5-advprop-autoaug': ModelTypes.ROBUST_INTV,
'efficientnet-b5-autoaug': ModelTypes.STANDARD,
'efficientnet-b5-randaug': ModelTypes.STANDARD,
'efficientnet-b6-advprop-autoaug': ModelTypes.ROBUST_INTV,
'efficientnet-b6-autoaug': ModelTypes.STANDARD,
'efficientnet-b7-advprop-autoaug': ModelTypes.ROBUST_INTV,
'efficientnet-b7-autoaug': ModelTypes.STANDARD,
'efficientnet-b7-randaug': ModelTypes.STANDARD,
'efficientnet-b8-advprop-autoaug': ModelTypes.ROBUST_INTV,
'efficientnet-l2-noisystudent': ModelTypes.MORE_DATA,
'facebook_adv_trained_resnet152_baseline': ModelTypes.LP_ADV,
'facebook_adv_trained_resnet152_denoise': ModelTypes.LP_ADV,
'facebook_adv_trained_resnext101_denoiseAll': ModelTypes.LP_ADV,
'fbresnet152': ModelTypes.STANDARD,
'google_resnet101_jft-300M': ModelTypes.MORE_DATA,
'googlenet/inceptionv1': ModelTypes.STANDARD,
'inceptionresnetv2': ModelTypes.STANDARD,
'inceptionv3': ModelTypes.STANDARD,
'inceptionv4': ModelTypes.STANDARD,
'instagram-resnext101_32x16d': ModelTypes.MORE_DATA,
'instagram-resnext101_32x32d': ModelTypes.MORE_DATA,
'instagram-resnext101_32x48d': ModelTypes.MORE_DATA,
'instagram-resnext101_32x8d': ModelTypes.MORE_DATA,
'mnasnet0_5': ModelTypes.STANDARD,
'mnasnet1_0': ModelTypes.STANDARD,
'mobilenet_v2': ModelTypes.STANDARD,
'mobilenet_v2_lpf2': ModelTypes.ROBUST_INTV,
'mobilenet_v2_lpf3': ModelTypes.ROBUST_INTV,
'mobilenet_v2_lpf5': ModelTypes.ROBUST_INTV,
'nasnetalarge': ModelTypes.STANDARD,
'nasnetamobile': ModelTypes.STANDARD,
'pnasnet5large': ModelTypes.STANDARD,
'polynet': ModelTypes.STANDARD,
'resnet101': ModelTypes.STANDARD,
'resnet101-tencent-ml-images': ModelTypes.MORE_DATA,
'resnet101_cutmix': ModelTypes.ROBUST_INTV,
'resnet101_lpf2': ModelTypes.ROBUST_INTV,
'resnet101_lpf3': ModelTypes.ROBUST_INTV,
'resnet101_lpf5': ModelTypes.ROBUST_INTV,
'resnet152': ModelTypes.STANDARD,
'resnet152-imagenet11k': ModelTypes.MORE_DATA,
'resnet152_3x_simclrv2_linear_probe_tf_port': ModelTypes.STANDARD,
'resnet152_3x_simclrv2_finetuned_100pct_tf_port': ModelTypes.STANDARD,
'resnet18': ModelTypes.STANDARD,
'resnet18-rotation-nocrop_40': ModelTypes.ROBUST_INTV,
'resnet18-rotation-random_30': ModelTypes.ROBUST_INTV,
'resnet18-rotation-random_40': ModelTypes.ROBUST_INTV,
'resnet18-rotation-standard_40': ModelTypes.ROBUST_INTV,
'resnet18-rotation-worst10_30': ModelTypes.ROBUST_INTV,
'resnet18-rotation-worst10_40': ModelTypes.ROBUST_INTV,
'resnet18_lpf2': ModelTypes.ROBUST_INTV,
'resnet18_lpf3': ModelTypes.ROBUST_INTV,
'resnet18_lpf5': ModelTypes.ROBUST_INTV,
'resnet18_ssl': ModelTypes.MORE_DATA,
'resnet18_swsl': ModelTypes.MORE_DATA,
'resnet34': ModelTypes.STANDARD,
'resnet34_lpf2': ModelTypes.ROBUST_INTV,
'resnet34_lpf3': ModelTypes.ROBUST_INTV,
'resnet34_lpf5': ModelTypes.ROBUST_INTV,
'resnet50': ModelTypes.STANDARD,
'resnet50-randomized_smoothing_noise_0.00': ModelTypes.STANDARD,
'resnet50-randomized_smoothing_noise_0.25': ModelTypes.LP_ADV,
'resnet50-randomized_smoothing_noise_0.50': ModelTypes.LP_ADV,
'resnet50-randomized_smoothing_noise_1.00': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_0.25': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_0.50': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_1.00': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_PGD_1step_eps_512_noise_0.25': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_PGD_1step_eps_512_noise_0.50': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_PGD_1step_eps_512_noise_1.00': ModelTypes.LP_ADV,
'resnet50-vtab': ModelTypes.STANDARD,
'resnet50-vtab-exemplar': ModelTypes.STANDARD,
'resnet50-vtab-rotation': ModelTypes.STANDARD,
'resnet50-vtab-semi-exemplar': ModelTypes.STANDARD,
'resnet50-vtab-semi-rotation': ModelTypes.STANDARD,
'resnet50_adv-train-free': ModelTypes.LP_ADV,
'resnet50_augmix': ModelTypes.ROBUST_INTV,
'resnet50_aws_baseline': ModelTypes.STANDARD,
'resnet50_clip_zeroshot': ModelTypes.MORE_DATA,
'resnet50_cutmix': ModelTypes.ROBUST_INTV,
'resnet50_cutout': ModelTypes.ROBUST_INTV,
'resnet50_deepaugment': ModelTypes.ROBUST_INTV,
'resnet50_deepaugment_augmix': ModelTypes.ROBUST_INTV,
'resnet50_feature_cutmix': ModelTypes.ROBUST_INTV,
'resnet50_imagenet_100percent_batch64_original_images': ModelTypes.STANDARD,
'resnet50_imagenet_subsample_125_classes_batch64_original_images': ModelTypes.LOW_ACCURACY_CNN,
'resnet50_imagenet_subsample_1_of_16_batch64_original_images': ModelTypes.LOW_ACCURACY_CNN,
'resnet50_imagenet_subsample_1_of_2_batch64_original_images': ModelTypes.LOW_ACCURACY_CNN,
'resnet50_imagenet_subsample_1_of_32_batch64_original_images': ModelTypes.LOW_ACCURACY_CNN,
'resnet50_imagenet_subsample_1_of_4_batch64_original_images': ModelTypes.LOW_ACCURACY_CNN,
'resnet50_imagenet_subsample_1_of_8_batch64_original_images': ModelTypes.LOW_ACCURACY_CNN,
'resnet50_imagenet_subsample_250_classes_batch64_original_images': ModelTypes.LOW_ACCURACY_CNN,
'resnet50_imagenet_subsample_500_classes_batch64_original_images': ModelTypes.LOW_ACCURACY_CNN,
'resnet50_l2_eps3_robust': ModelTypes.LP_ADV,
'resnet50_linf_eps4_robust': ModelTypes.LP_ADV,
'resnet50_linf_eps8_robust': ModelTypes.LP_ADV,
'resnet50_lpf2': ModelTypes.ROBUST_INTV,
'resnet50_lpf3': ModelTypes.ROBUST_INTV,
'resnet50_lpf5': ModelTypes.ROBUST_INTV,
'resnet50_mixup': ModelTypes.ROBUST_INTV,
'resnet50_simclrv2_linear_probe_tf_port': ModelTypes.STANDARD,
'resnet50_simclrv2_finetuned_100pct_tf_port': ModelTypes.STANDARD,
'resnet50_simsiam': ModelTypes.STANDARD,
'resnet50_ssl': ModelTypes.MORE_DATA,
'resnet50_swav': ModelTypes.STANDARD,
'resnet50_swsl': ModelTypes.MORE_DATA,
'resnet50_trained_on_SIN': ModelTypes.ROBUST_INTV,
'resnet50_trained_on_SIN_and_IN': ModelTypes.ROBUST_INTV,
'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': ModelTypes.ROBUST_INTV,
'resnet50_with_brightness_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_contrast_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_defocus_blur_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_fog_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_frost_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_gaussian_noise_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_gaussian_noise_contrast_motion_blur_jpeg_compression_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_greyscale_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_jpeg_compression_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_motion_blur_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_pixelate_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_saturate_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_spatter_aws': ModelTypes.ROBUST_INTV,
'resnet50_with_zoom_blur_aws': ModelTypes.ROBUST_INTV,
'resnext101_32x16d_ssl': ModelTypes.MORE_DATA,
'resnext101_32x4d': ModelTypes.STANDARD,
'resnext101_32x4d_ssl': ModelTypes.MORE_DATA,
'resnext101_32x4d_swsl': ModelTypes.MORE_DATA,
'resnext101_32x8d': ModelTypes.STANDARD,
'resnext101_32x8d_deepaugment_augmix': ModelTypes.ROBUST_INTV,
'resnext101_32x8d_ssl': ModelTypes.MORE_DATA,
'resnext101_32x8d_swsl': ModelTypes.MORE_DATA,
'resnext101_64x4d': ModelTypes.STANDARD,
'resnext50_32x4d': ModelTypes.STANDARD,
'resnext50_32x4d_ssl': ModelTypes.MORE_DATA,
'resnext50_32x4d_swsl': ModelTypes.MORE_DATA,
'se_resnet101': ModelTypes.STANDARD,
'se_resnet152': ModelTypes.STANDARD,
'se_resnet50': ModelTypes.STANDARD,
'se_resnext101_32x4d': ModelTypes.STANDARD,
'se_resnext50_32x4d': ModelTypes.STANDARD,
'senet154': ModelTypes.STANDARD,
'shufflenet_v2_x0_5': ModelTypes.STANDARD,
'shufflenet_v2_x1_0': ModelTypes.STANDARD,
'squeezenet1_0': ModelTypes.STANDARD,
'squeezenet1_1': ModelTypes.STANDARD,
'vit_b_32_clip_zeroshot': ModelTypes.MORE_DATA,
'vgg11': ModelTypes.STANDARD,
'vgg11_bn': ModelTypes.STANDARD,
'vgg13': ModelTypes.STANDARD,
'vgg13_bn': ModelTypes.STANDARD,
'vgg16': ModelTypes.STANDARD,
'vgg16_bn': ModelTypes.STANDARD,
'vgg16_bn_lpf2': ModelTypes.ROBUST_INTV,
'vgg16_bn_lpf3': ModelTypes.ROBUST_INTV,
'vgg16_bn_lpf5': ModelTypes.ROBUST_INTV,
'vgg16_lpf2': ModelTypes.ROBUST_INTV,
'vgg16_lpf3': ModelTypes.ROBUST_INTV,
'vgg16_lpf5': ModelTypes.ROBUST_INTV,
'vgg19': ModelTypes.STANDARD,
'vgg19_bn': ModelTypes.STANDARD,
'vit_small_patch16_224': ModelTypes.STANDARD,
'vit_base_patch16_224': ModelTypes.MORE_DATA,
'vit_base_patch16_384': ModelTypes.MORE_DATA,
'vit_base_patch32_384': ModelTypes.MORE_DATA,
'vit_large_patch16_224': ModelTypes.MORE_DATA,
'vit_large_patch16_384': ModelTypes.MORE_DATA,
'vit_large_patch32_384': ModelTypes.MORE_DATA,
'wide_resnet101_2': ModelTypes.STANDARD,
'wide_resnet50_2': ModelTypes.STANDARD,
'xception': ModelTypes.STANDARD,
'resnet50_lstsq': ModelTypes.RANDOM_FEATURES,	
'identity32_lstsq': ModelTypes.LINEAR_PIXELS,	
'identity32_random_forests': ModelTypes.RANDOM_FORESTS,
'identity32_one_nn': ModelTypes.NEAREST_NEIGHBORS
}

for i in range(100):
    model_types_map[f"resnet18_50k_{i}_epochs"] =  ModelTypes.LOW_ACCURACY_CNN

for i in range(50):
    model_types_map[f"resnet18_100k_{i}_epochs"] =  ModelTypes.LOW_ACCURACY_CNN

for i in range(10):
    model_types_map[f"resnet101_{i}_epochs"] =  ModelTypes.LOW_ACCURACY_CNN

class NatModelTypes(Enum):
    STANDARD = ('Standard training', 'tab:blue')
    ROBUST_INTV = ('Robustness intervention', 'tab:brown')
    MORE_DATA = ('Trained with more data', 'tab:green')
    RANDOM_FEATURES = ('Random Features', 'tab:olive')	
    LINEAR_PIXELS = ('Linear Classifier on Pixels', 'tab:orange')	
    RANDOM_FORESTS = ('Random Forests', 'tab:pink')
    NEAREST_NEIGHBORS = ('Nearest Neighbors', 'tab:purple')
    LOW_ACCURACY_CNN = ('Low Accuracy CNN', 'tab:cyan')

mapper = {
	ModelTypes.STANDARD: NatModelTypes.STANDARD,
	ModelTypes.LP_ADV: NatModelTypes.ROBUST_INTV,
	ModelTypes.ROBUST_INTV: NatModelTypes.ROBUST_INTV,
	ModelTypes.MORE_DATA: NatModelTypes.MORE_DATA,	
	ModelTypes.RANDOM_FEATURES: NatModelTypes.RANDOM_FEATURES,	
	ModelTypes.LINEAR_PIXELS: NatModelTypes.LINEAR_PIXELS,	
	ModelTypes.RANDOM_FORESTS: NatModelTypes.RANDOM_FORESTS,	
	ModelTypes.NEAREST_NEIGHBORS : NatModelTypes.NEAREST_NEIGHBORS,
	ModelTypes.LOW_ACCURACY_CNN : NatModelTypes.LOW_ACCURACY_CNN
}
nat_model_types_map = {k: mapper[v] for k, v in model_types_map.items()}
