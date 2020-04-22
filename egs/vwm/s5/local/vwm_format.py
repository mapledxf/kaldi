import sys,os
import re
from shutil import copyfile
import number_converter
import difflib

def show_diff(seqm):
    """Unify operations between two compared strings
seqm is a difflib.SequenceMatcher instance whose a & b are strings"""
    output= []
    for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
        if opcode == 'equal':
            output.append(seqm.a[a0:a1])
        elif opcode == 'insert':
            output.append("<ins>" + seqm.b[b0:b1] + "</ins>")
        elif opcode == 'delete':
            output.append("<del>" + seqm.a[a0:a1] + "</del>")
        elif opcode == 'replace':
            output.append("<rep>" + seqm.b[b0:b1] + "</rep>")
    return ''.join(output)

vwm_dir=sys.argv[1]
out_dir=sys.argv[2]

special = open('./local/vwm_special.txt')
special_list = []
#list of trans need to use inverted number
for line in special:
    line = line.strip('\r\n')
    if line:
        special_list.append(line)

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

        trans = trans.replace(':','点')
        # convert numbers to chinese
        split = re.findall(r'[A-Za-z\s]+|[\u4e00-\u9fcc]+|-?\d+\.\d+|\d+', trans)
        result = ""
        priv = ''
        nex = ''
        for i in range(len(split)):
            word = split[i].replace(' '," ")
            if not word.replace(" ",""):
                continue
            if i < len(split)-1:
                nex = split[i+1][0]
            else:
                nex = ''

            if number_converter.is_number(word):
                unit = True
                if word.isdigit():
                    if len(word) == 1:
                        unit = True
                    else:
                        # 火车号或者航班号
                        if word.startswith('0'):
                            unit = False
                        elif priv.encode('utf-8').isalpha():
                            unit = False
                        elif nex in {'年'}:
                            unit = False
                        elif priv in {'点','月','第'}:
                            unit = True
                        elif nex in {'幢','栋','时','路','元','块','米','以','日','号','点','月','元','弄','天','度'}:
                            unit = True
                        elif word.endswith('0'):
                            unit = True
                        elif word == '24':
                            unit = True
                        elif len(word) == 2:
                            unit = True
                        elif len(word) == 3:
                            unit = False
                        elif len(word) == 4:
                            unit = False
                        else:
#                            print(word + " in " + trans.strip("\r\n") + " " + priv + " " + nex + " " + wav)
                            unit = False
                    if wav in special_list:
                        unit = not unit
                result +=number_converter.num2chinese(word, unit=unit)
            else:
                result +=word
            priv = word[-1:]
#        if result != trans:
#            sm= difflib.SequenceMatcher(None, trans, result)
#            print(wav + "\t" + show_diff(sm))

        new_script.write(new_name + " " + result)
    else:
        print (old_path + " not exist")

script.close()
new_script.close()
special.close()

print ("All done!")
