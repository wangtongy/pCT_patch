import os
# index_model=208

# os.mkdir("/data/anlab/TongyaoW/BlackBoneProject/Data/MR2CT/3D_Dataset/outputs_test_g1/%d"%(index_model))

model = "/data/anlab/TongyaoW/BlackBoneProject/PretrainedModel/NetBM/t_2001.pt"
image = "/data/anlab/TongyaoW/BlackBoneProject/Data/MR2CT/3D_Dataset/test_single_subject/SB_007"
output = "/data/anlab/TongyaoW/BlackBoneProject/pCT_Code_clean/pCT_BM"
script_file_name = '/data/anlab/TongyaoW/BlackBoneProject/pCT_Code_clean/BM/runTesting_Recon.py'
argument = []
argument.append('--using_r1 1')
argument.append('--using_r2 0')
argument.append('--using_fa3e1 0')
argument.append('--using_fa15e1 0')
argument.append('--using_fa15e2 0')
argument.append('--using_t1 0')
argument.append('--modelPath'+' '+model)
argument.append('--image_path'+' '+image)
argument.append('--output_path'+' '+output)
argument.append('--gpuID 1')
argument.append('--step1 16')
argument.append('--step2 16')
argument.append('--step3 16')
argument_all = ' '.join(argument)
command = ' '.join(['python', script_file_name, argument_all])
os.system(command)