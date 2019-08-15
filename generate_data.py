import dlutils

source_path_A = '/media/antonio/Data/DataSets/Raw/Selfie-dataset/images'
source_path_B = '/media/antonio/Data/DataSets/Raw/LLD-logo-files'
destination_path = '/media/antonio/Data/DataSets/Projects/Stickerizer/FaceToSticker'

dlutils.generate_dataset_GAN(
    source_path_A, source_path_B, destination_path, 0.2, 0.1)
