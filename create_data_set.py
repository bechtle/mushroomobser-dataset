import os
import shutil
import operator
import sys

def create_validation_set(species_list,num_classes,drive_path_validate):
    drive_path_validate_num_classes='VALIDATE_'+str(num_classes)+'/'
    if os.path.isdir(drive_path_validate_num_classes)==False:
        os.makedirs(drive_path_validate_num_classes)
    
    species = dict(species_list[:num_classes])
    spec_dirs = os.listdir(drive_path_validate)
    for spec_dir in spec_dirs:
        if spec_dir in species:
            shutil.copytree(drive_path_validate+spec_dir, drive_path_validate_num_classes+spec_dir)


def create_train_set(num_classes,drive_path_train,drive_path_validate):
    
    folder_dict={}
    spec_dirs = os.listdir(drive_path_train)[1:]
    for spec_dir in spec_dirs:
        files = os.listdir(drive_path_train+spec_dir)
        folder_dict[spec_dir] = len(files)

    sorted_x = sorted(folder_dict.items(), key=operator.itemgetter(1))[::-1]

    species_list = []
    for i in sorted_x:
        name=i[0]
        name_comp = name.split(' ')
        if len(name_comp)==2:
            if name_comp[0][0].isupper()==True and name_comp[1][0].isupper()==False:
                species_list.append(i)
    
    
    drive_path_train_num_classes='TRAIN_'+str(num_classes)+'/'
    if os.path.isdir(drive_path_train_num_classes)==False:
        os.makedirs(drive_path_train_num_classes)
        
    species = dict(species_list[:num_classes])
    spec_dirs = os.listdir(drive_path_train)[1:]
    for spec_dir in spec_dirs:
        if spec_dir in species:
            shutil.copytree(drive_path_train+spec_dir, drive_path_train_num_classes+spec_dir)
    
    create_validation_set(species_list,num_classes,drive_path_validate)



num_classes = int(sys.argv[1])
drive_path_train=sys.argv[2]
drive_path_validate=sys.argv[3]

create_train_set(num_classes,drive_path_train,drive_path_validate)