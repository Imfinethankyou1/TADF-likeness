
import glob




#for ratio in [0.25]:
def screening(lines):
    new_lines = []
    for line in lines:
        tmp = line.strip().split()
        #print(tmp)
        est = float(tmp[-2]) - float(tmp[-1])
        if 0 < est<0.4 and float(tmp[-2]) > 2.0 and float(tmp[-2]) < 3.0:
            new_lines.append(line)

    new_lines = list(set(new_lines))
    new_lines.sort(key=len)
    return new_lines



def analysis(filenames, ratio_on=False,f_write=False):

#if True:
    if ratio_on:
        for ratio in [0.75, 0.5, 0.25, 0.20, 0.15, 0.1, 0.05, 0.0]:
            lines = []
            for filename in filenames:
                with open(filename) as f:
                    lines += f.readlines()
            new_lines = screening(lines)
            print(ratio , len(new_lines), len(lines))
    else:
        lines = []
        for filename in filenames:
            with open(filename)  as f:
                lines += f.readlines()
        new_lines = screening(lines)            

    if f_write:
        with open('data_220615/EST_pub_chromophore.txt','w') as f:
            for line in new_lines:
                f.write(line)
    #for line in new_lines:
        #   print(line, end ='')
    

if __name__ =='__main__':
    
    #filenames = [f'output/origin_threshold_{ratio}_TADF_candidates_from_Pub.txt']
    f_write = True
    ratio_on = False    
    filenames = ['data_220615/test_chromophore.txt','data_220615/total_pub.txt' ] 
    analysis(filenames, ratio_on, f_write)
