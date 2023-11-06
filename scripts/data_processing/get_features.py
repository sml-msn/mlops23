import sys 
import os
import io

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython get_features.py data-file\n")
    sys.exit(1)
    
f_input = sys.argv[1]
f_output = os.path.join("datasets", "stage1", "train.csv")
os.makedirs(os.path.join("datasets", "stage1"), exist_ok=True)

def process_data(fd_in, fd_out):
    fd_in.readline()
    for line in fd_in:
        line = line.rstrip('\n').split(',')
        p_survived = line[2] 
        p_pclass = line[3]
        if line[4][0] == '"':
            p_sex = line[6]
            p_age = line[7]
        else:
            p_sex = line[5]
            p_age = line[6]
        fd_out.write('{},{},{},{}\n'.format(p_survived, p_pclass, p_sex, p_age))

with io.open(f_input, encoding="utf8") as fd_in:
    with io.open(f_output, 'w', encoding='utf8') as fd_out:
        process_data(fd_in, fd_out)
