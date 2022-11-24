%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/142583504_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/142583504_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.0164216659        0.5254358237        0.1127184764
C                 4.0118309833       -0.3123526434        0.9240747839
C                 2.6171983060       -0.3383714335        0.3220254906
C                 2.1766964593       -1.4597717940       -0.2723788358
C                 0.8388527544       -1.7530953141       -0.9184247113
C                 0.2458378957       -0.6336473183       -1.7899260580
N                -0.1188385967        0.5670374710       -1.0509471984
C                -1.3315982202        0.6848347188       -0.4020108913
C                -2.3845406036       -0.2467880440       -0.5825144647
C                -3.5952667705       -0.0378363714        0.0626089734
C                -4.6734376090       -1.0684786596       -0.1581636524
N                -5.7107995619       -1.0534309662        0.7413257500
O                -4.6374902749       -1.8587404263       -1.0925275459
C                -3.7559946534        1.0961557330        0.8753095803
C                -2.6617718953        1.9445112212        1.0036759712
N                -1.4823485888        1.7543857714        0.4088517725
C                 0.9893839208        1.4456046496       -0.6861833563
C                 1.8221580025        0.9468312654        0.5057166070
H                 5.1429350712        0.1103961662       -0.8934518711
H                 5.9973557047        0.5394325548        0.6016922410
H                 4.6867675970        1.5655124850        0.0049864624
H                 4.3941028191       -1.3359844288        1.0160850507
H                 3.9569134841        0.0926897435        1.9457202893
H                 2.8617087537       -2.3085039372       -0.2695175397
H                 0.9469950764       -2.6443277490       -1.5503479723
H                 0.1007765189       -2.0246486972       -0.1474883219
H                 0.9803177337       -0.3381092511       -2.5482473597
H                -0.6269864524       -1.0063517166       -2.3327005765
H                -2.2922751483       -1.1251411896       -1.2069344051
H                -5.5917056646       -0.6376351954        1.6534618706
H                -6.3612501271       -1.8247709647        0.6722365585
H                -4.6948766539        1.3350627616        1.3631217873
H                -2.7322308017        2.8350849490        1.6275458823
H                 0.5746918913        2.4242216336       -0.4436316487
H                 1.6298820503        1.5498499372       -1.5702669221
H                 2.5130085205        1.7554603476        0.7779227345
H                 1.1475764134        0.8354788676        1.3673640489


