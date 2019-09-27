
import os
import shutil
import cv2


def generate_datasets_CNN(source, destination, test_factor, validation_factor):
    file_names = os.listdir(source)
    file_names = sorted(file_names)
    total_files = len(file_names)

    num_train_files = int(
        total_files * (1.0 - test_factor - validation_factor))
    num_test_files = int(total_files * test_factor)
    num_validation_file = int(total_files * validation_factor)

    train_dir = os.path.join(destination, 'train')
    os.mkdir(train_dir)
    test_dir = os.path.join(destination, 'test')
    os.mkdir(test_dir)
    validation_dir = os.path.join(destination, 'validation')
    os.mkdir(validation_dir)

    for i in range(num_train_files):
        file_name = file_names[i]
        complete_source = os.path.join(source, file_name)
        complete_destination = os.path.join(train_dir, file_name)
        shutil.copy(complete_source, complete_destination)

    for i in range(num_train_files, num_train_files + num_test_files):
        file_name = file_names[i]
        complete_source = os.path.join(source, file_name)
        complete_destination = os.path.join(test_dir, file_name)
        shutil.copy(complete_source, complete_destination)

    for i in range(num_train_files + num_test_files, num_train_files + num_test_files + num_validation_file):
        file_name = file_names[i]
        complete_source = os.path.join(source, file_name)
        complete_destination = os.path.join(validation_dir, file_name)
        shutil.copy(complete_source, complete_destination)


def generate_dataset_GAN(source_domainA, source_domainB, target, test_factor, validation_factor):
    domain = os.path.basename(os.path.normpath(target))
    destination_domainA = os.path.join(target, '{}_A'.format(domain))
    os.mkdir(destination_domainA)

    generate_datasets_CNN(source_domainA, destination_domainA,
                          test_factor, validation_factor)

    destination_domainB = os.path.join(target, '{}_B'.format(domain))
    os.mkdir(destination_domainB)
    generate_datasets_CNN(source_domainB, destination_domainB,
                          test_factor, validation_factor)


def downscale_upscale_folders(path, down_path='/downscaled', resize_path='/resized', factor=4):

    downscale_path = os.path.join(path, down_path)
    resize_final_path = os.path.join(path, resize_path)
    if not os.path.exists(downscale_path):
        os.makedirs(downscale_path)
    if not os.path.exists(resize_final_path):
        os.makedirs(resize_final_path)
    total_files = len(os.listdir(path))
    i = 0
    for filename in os.listdir(path):
        i += 1
        print('File {} {} / {}'.format(filename, i, total_files))
        im_ori = cv2.imread(os.path.join(path, filename))
        base_name = os.path.basename(filename)
        base_name, ext = base_name.split('.')
        final_name = base_name + '_downscaled.' + ext
        final_path = os.path.join(downscale_path, final_name)
        im_downscaled = cv2.resize(
            im_ori, (im_ori.shape[0]//factor, im_ori.shape[1]//factor))
        cv2.imwrite(final_path, im_downscaled)
        im_upscalled = cv2.resize(
            im_downscaled, (im_ori.shape[0], im_ori.shape[1]))
        final_name = base_name + '_resized.' + ext
        final_path = os.path.join(resize_final_path, final_name)
        cv2.imwrite(final_path, im_upscalled)
