#created by Mattias Heinrich

import os
import subprocess
#from subprocess import call

list_of_labels = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48]
#list_of_labels = [2,3,4,5,6,7,8,9,10,11,12,13,14,]

for i in range(len(list_of_labels)):
    name = 'PANCREAS_'+str(list_of_labels[i]).zfill(4)
    print('name',name)
    dict0 = 'DOI/'+name
    dict1 = next(os.walk(dict0))[1][0]

    dict1 = os.path.join(dict0, dict1)
    dict2 = next(os.walk(dict1))[1][0]
    dict2 = os.path.join(dict1, dict2)



    #we first need to obtain a dicom series id
    list_command = './c3d/bin/c3d -dicom-series-list '+dict2
    proc = subprocess.Popen([list_command], stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    p_status = proc.wait()

    line = (out.splitlines()[1])
    series_id = str.split(line)[-1]
    print('reading series_id',series_id)

    #next we apply the dicom read and convert function
    read_command = './c3d/bin/c3d -dicom-series-read '+dict2+' '+series_id+' -o '+name+'.nii.gz'
    proc = subprocess.Popen([read_command], stdout=subprocess.PIPE, shell=True)
    p_status = proc.wait()

    #subprocess.call(read_command)
    print(err)
    print(out)
    print(name+' has been converted to nii.gz')

    #check number of zero voxels
    count_command = 'c3d/bin/c3d '+name+'.nii.gz -thresh 0 0 1 0 -voxel-sum'
    proc = subprocess.Popen([count_command], stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    p_status = proc.wait()

    print(out)