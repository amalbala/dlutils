
import os
import shutil


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
