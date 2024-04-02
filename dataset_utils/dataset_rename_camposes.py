import os

def rename_files(folder_path):
    

    for filename in os.listdir(folder_path):
    # print(filename)
    #     new_filename = filename.replace(f'{i}', f'{new_i}')
    #     
    #     print(os.path.join(folder_path, new_filename))
    #     j += 1
    

        filename_rename = filename.split('_')
       
        new_filename = f'cam_poses_{int(filename_rename[2])+200}_{filename_rename[3]}'
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        print(f'done_{os.path.join(folder_path, new_filename)}')

if __name__ == '__main__':
    
    folder_path = '/home/cyf/task_grasp/A-G/datasets/rotate_valve/1'
    
    rename_files(folder_path)
    
    print('done')