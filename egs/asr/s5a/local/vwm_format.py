import sys,os
import re
from shutil import copyfile

vwm_dir=sys.argv[1]
out_dir=sys.argv[2]

vwm_script = os.path.join(vwm_dir, 'script')
out_script = os.path.join(out_dir, 'script')
if(not os.path.isdir(out_script)):
    os.makedirs(out_script)

vwm_wav_dir = os.path.join(vwm_dir,'wav')
out_wav_dir = os.path.join(out_dir,'wav')
if(not os.path.isdir(out_wav_dir)):
    os.makedirs(out_wav_dir)

script = open(os.path.join(vwm_script,'script.txt'))
new_script = open(os.path.join(out_script, 'script.txt'),"w")

for line in script.readlines():
    wav,trans = line.replace(' ','\t').split('\t',1)
    s = wav.split('_')[1:]
    s[0]=s[0].zfill(4)
    new_name = "_".join(s)

    new_path = os.path.join(out_wav_dir, new_name)
    old_path = os.path.join(vwm_wav_dir, wav)
    if os.path.isfile(old_path):
        copyfile(old_path, new_path)
        new_script.write(new_name + " " + trans.replace(' ',' '))
    else:
        print (old_path + " not exist")

script.close()
new_script.close()

print ("All done!")
