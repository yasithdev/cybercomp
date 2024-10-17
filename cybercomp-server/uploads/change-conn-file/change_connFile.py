#--------------------------------
# Modify connection_info file instead of generating the whole network. 
# Useful for small changes, like scaling strength for a specific connection type.
#--------------------------------

# import subprocess
# import os
# Cell types: 
# RE     642  1  #0 
# REa    642  1  #1 
# TC     642  1  #2 
# TCa    642  1  #3 
# CX    10242  1  #4 
# CX3    10242  1  #5 
# CX4    10242  1  #6 
# CX5a   10242  1  #7 
# CX5b   10242  1  #8 
# CX6    10242  1  #9 
# IN    10242  1  #10 
# IN3    10242  1  #11 
# IN4    10242  1  #12 
# IN5a   10242  1  #13 
# IN5b   10242  1  #14 
# IN6    10242  1  #15 

# cwd = os.getcwd()
# Relevant cell and synapse types to modify (note: range() EXCLUDES last element)
# pre_types = [str(x) for x in range(4,10)] # pyramidal source cell types 
# post_types = [str(x) for x in range(0,4)] # thalamic target cell types
# syn_types = ['AMPA'] # synapse types
# pre_types = [str(x) for x in range(0,2)] # RE source cell types 
# post_types = [str(x) for x in range(0,2)] # RE target cell types
# syn_types = ['AMPA'] # synapse types
# pre_types = str(3) # TCa matrix source cell types 
# post_types = [str(x) for x in range(4,10)] # pyramidal target cell types

variant = 'nothcx' # name of the modification
pre_types = [str(x) for x in range(0,4)] # thalamic source cell types
post_types = [str(x) for x in range(4,16)] # all cortical (CX/IN) target cell types

# variant = 'cxcx_0_49' # name of the modification
# pre_types = [str(x) for x in range(4,10)] # pyramidal source cell types 
# post_types = [str(x) for x in range(4,16)] # all cortical (CX/IN) target cell types
# syn_types = ['AMPAMap_D1'] # synapse types

f_in = "connection_info_full_cxth_10_cxcx_0_49"
f_out = f_in + '_' + variant
# Check differences
f_diff = 'diff_' + f_in + "_v_" + variant

with open(f_in, 'r') as file_in, open(f_out, 'w') as file_out, open(f_diff, 'w') as file_diff:
    read_cell = 0 #flag for when connection list for relevant cell begins in .cfg
    for line in file_in:
        # File fields:
        # In: to_t  to_x  (to_y)
        # 0:from_t 1:from_x  2:(from_y) 3:synapse  4:strength*weightFactor  5:mini_strength 6:mini_freq  7:range  8:delay
        fields = line.strip().split()
        if fields:
            if fields[0]=='In:' : # new cell connection list begins
                if fields[1] in post_types: # relevant cell: read
                    read_cell = 1
                    file_diff.writelines(line)
                else: # irrelevant cell: ignore
                    read_cell = 0
            else:
                # if read_cell and (fields[0] in pre_types) and (fields[3] in syn_types):
                if read_cell and (fields[0] in pre_types):
                    file_diff.writelines('-' + line)
                    #print('Old:', fields)
                    fields[4]= str(float(fields[4])*0) # modify value 
                    line = ' '.join(fields)+'\n'
                    file_diff.writelines('+' + line)
                    #print('New:', fields)   
        file_out.writelines(line)

# print("File created, writing diffs... ")

# # Check differences
# f_diff = 'diff_' + variant
# # subprocess.call(["diff", "-u", f_in, f_out, " > ", f_diff])
# res = subprocess.check_output(["diff", "-u", cwd + "/" + f_in, cwd +"/" + f_out])
# with open(f_diff, 'w') as file_diff:
#     for line in res.splitlines():
#         # process the output line by line
#         file_diff.writelines(line)