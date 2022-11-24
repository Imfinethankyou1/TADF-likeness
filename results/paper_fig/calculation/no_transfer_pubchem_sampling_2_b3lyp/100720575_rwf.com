%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/100720575_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/100720575_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -4.5636534870        3.0264967242       -0.6743459171
C                -3.8803405544        2.1918418930        0.4049725102
O                -2.9378189332        3.0507921151        1.0601404688
C                -3.1734756059        0.9682349481       -0.1257101269
C                -2.3160071027        0.8561272036       -1.1884184924
C                -1.6018130489       -0.3777966100       -1.2100321748
C                -1.9017032306       -1.2146247560       -0.1629683941
C                -1.1771690672       -2.4617548273        0.3048198076
C                -0.1964401893       -2.0710190198        1.4529065201
N                 0.7871749703       -1.0616258583        1.0983475433
C                 0.6350498381        0.2719098139        1.3972893559
O                -0.2929441027        0.7095895070        2.0771237354
C                 1.6874612912        1.1908880574        0.8798853340
C                 2.9710541301        0.8596914991        0.6501923780
C                 4.0536608580        1.7430175652        0.2004593653
C                 3.8551551921        3.0962236070       -0.1337306884
C                 4.9182579029        3.8878657374       -0.5555376030
C                 6.2052616008        3.3496229680       -0.6563724203
C                 6.4190591667        2.0098774967       -0.3313069149
C                 5.3533656351        1.2174194113        0.0913690671
C                -2.1529027315       -3.5722419422        0.7746815499
C                -2.6902992256       -4.2080695421       -0.5257934587
C                -1.5795423235       -3.9780936845       -1.5928929822
C                -0.4667827254       -3.1892120429       -0.8654700387
S                -3.1345470123       -0.4832493196        0.8388706529
H                -5.3571127930        2.4541477309       -1.1653579199
H                -3.8400532686        3.3500693701       -1.4292994358
H                -5.0003817664        3.9215249267       -0.2202737025
H                -4.6458199423        1.8662005369        1.1287221984
H                -2.2914173481        2.4859199768        1.5189563413
H                -2.1629783951        1.6482782321       -1.9137374446
H                -0.8602399990       -0.6127152654       -1.9660727947
H                -0.7556700177       -1.6731765559        2.3024150159
H                 0.3310220254       -2.9754433549        1.7842908470
H                 1.4535547191       -1.2948079628        0.3759296029
H                 1.3392405920        2.2156040302        0.7880974327
H                 3.2811526346       -0.1692006819        0.8355321597
H                 2.8625648583        3.5310914909       -0.0652393179
H                 4.7439104659        4.9301978582       -0.8084243258
H                 7.0326867290        3.9715872582       -0.9865459767
H                 7.4147585177        1.5812024895       -0.4059393917
H                 5.5255139543        0.1738939205        0.3457695880
H                -1.5838415442       -4.3126337991        1.3533134781
H                -2.9477562674       -3.1991681767        1.4290666220
H                -3.6176550306       -3.7130227019       -0.8310160194
H                -2.9250840682       -5.2684701702       -0.3876024845
H                -1.9723977939       -3.3978586646       -2.4335429088
H                -1.1960369884       -4.9171120808       -2.0053991204
H                 0.2645051130       -3.8870401393       -0.4344369931
H                 0.0874743683       -2.5209812120       -1.5321605270


