import os
import subprocess
#from subprocess import call
import pandas
#list_of_labels = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48]
#list_of_labels = [14,16]

df1 = pandas.read_csv('cropping.csv')
print(len(df1))
for i in range(1,43+1,1):#len(list_of_labels)):
    
    #Image #1: dim = [512, 512, 210];  bb = {[0 0 0], [439.296 439.296 210]};  vox = [0.858, 0.858, 1];  range = [-2048, 1371];  orient = RPS

    
    list_id = int(df1.loc[i-1,:]['original_id'])
    new_id = int(df1.loc[i-1,:]['id'])
    
    name_ct = 'PANCREAS_'+str(list_id).zfill(4)
    name_label = 'label'+str(list_id).zfill(4)
    print('name',name_ct,'label',name_label)
    
    info_command = './c3d/bin/c3d '+name_label+' -info'
    proc = subprocess.Popen([info_command], stdout=subprocess.PIPE, shell=True)
    line, err = proc.communicate()
    p_status = proc.wait()

    #print(line)
    dim_z = int(str.split(line)[6][:-2])
    print('dimension_str',dim_z)
    
    print(df1.loc[i-1,:])
    #print(df1.loc[i-1,:]['extent_ant'])
    
    A1 = int(df1.loc[i-1,:]['extent_ant'])
    A2 = int(df1.loc[i-1,:]['extent_post'])
    A3 = int(df1.loc[i-1,:]['extent_left'])
    A4 = int(df1.loc[i-1,:]['extent_right'])
    A5 = int(df1.loc[i-1,:]['extent_inf'])
    A6 = int(df1.loc[i-1,:]['extent_sup'])
    
#    region1 = str(A3)+'x'+str(512-A2)+'x'+str(A5)+'vox '+str(A4-A3)+'x'+str(A2-A1)+'x'+str(A6-A5)+'vox'
    region1 = str(A3)+'x'+str(512-A2)+'x'+str(dim_z-A6)+'vox '+str(A4-A3)+'x'+str(A2-A1)+'x'+str(A6-A5)+'vox'
    print('region',region1)
    
    label_command = './c3d/bin/c3d '+name_label+'.nii.gz -region '+region1+' -int 0 -resample 144x144x144 -type short -o label_ct'+str(new_id)+'.nii.gz'

    #we first need to obtain a dicom series id
    #list_command = './c3d/bin/c3d -dicom-series-list '+dict2
    proc = subprocess.Popen([label_command], stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    p_status = proc.wait()

    print(out)
    
    scan_command = './c3d/bin/c3d '+name_ct+'.nii.gz -type float -region '+region1+' -resample 144x144x144 -o pancreas_ct'+str(new_id)+'.nii.gz'

    proc = subprocess.Popen([scan_command], stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    p_status = proc.wait()

    print(out)
    
    print('pancreas_ct'+str(new_id)+'.nii.gz')
    
    #check number of zero voxels
    count_command = 'c3d/bin/c3d pancreas_ct'+str(new_id)+'.nii.gz -thresh 0 0 1 0 -voxel-sum'
    proc = subprocess.Popen([count_command], stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    p_status = proc.wait()

    print(out)
    
    #line = (out.splitlines()[1])
    #series_id = str.split(line)[-1]
    #print('reading series_id',series_id)


