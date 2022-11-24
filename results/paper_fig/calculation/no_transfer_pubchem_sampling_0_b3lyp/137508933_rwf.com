%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/137508933_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/137508933_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 0.3140720760       -1.5944400693        2.1810881174
N                 0.1273893399       -0.7714769926        0.9883748880
C                -1.0339867250       -0.6811619220        0.2576940099
N                -2.2386670940       -1.2914810001        0.4276640573
C                -3.0431527564       -0.8987283657       -0.5254352435
C                -4.4746188223       -1.3591753819       -0.6367868054
N                -5.4390286716       -0.3454571719       -0.2516210838
C                -5.8916990622       -0.0292172807        1.0327360382
C                -6.7319857112        1.0229654886        0.9200356935
N                -6.7898097283        1.3636843142       -0.4328276920
C                -5.9701417994        0.5409535009       -1.1885149688
O                -5.7409740317        0.5670568985       -2.3932533048
S                -2.3661832816        0.2592500797       -1.7001534835
C                -0.8714663225        0.1941798166       -0.8153852067
C                 0.4603947897        0.6802451748       -0.7530385737
C                 1.2690738582        1.5733216151       -1.5077010847
N                 2.5084333136        1.8354023083       -1.1925881334
N                 3.0340160782        1.2223026527       -0.1001868290
C                 4.4381301722        1.5822456446        0.1965317940
C                 5.4359607425        0.5376588301       -0.2006537822
C                 6.4071353335        0.5179552029       -1.1888679833
C                 7.0418249241       -0.7346444149       -1.0403935929
N                 6.5117665692       -1.4477932747       -0.0443708180
N                 5.5495746525       -0.6498327398        0.4550174693
C                 2.4071182401        0.3058575409        0.7483980780
O                 3.0188085997       -0.2036253847        1.7046721874
C                 1.0394289915        0.0595296336        0.3750758648
H                -0.3715244761       -1.2682890007        2.9690013382
H                 0.1063705257       -2.6413521861        1.9421841202
H                 1.3452437129       -1.4786328151        2.5097079106
H                -4.7076917753       -1.6356740841       -1.6690817344
H                -4.5930742780       -2.2366173018        0.0042150909
H                -5.5822008131       -0.5913144437        1.8998599701
H                -7.2981097874        1.5499141364        1.6713965933
H                -7.2997230459        2.1295838804       -0.8438917540
H                 0.8927658187        2.0883659281       -2.3864000196
H                 4.6251349042        2.5023955723       -0.3555636118
H                 4.4983276888        1.7872567852        1.2694641736
H                 6.6219192440        1.2984824725       -1.9049013045
H                 7.8611523007       -1.1471283616       -1.6148402030
H                 4.9299953063       -0.9633212847        1.1933438184


