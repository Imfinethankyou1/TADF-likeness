%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/115550172_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/115550172_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 0.8994466483        2.2359294998        0.7964426909
C                 1.5267740934        1.0758602098        0.0554428605
C                 2.8326821593        1.2122554499       -0.4354658658
C                 3.4585659297        0.1589558175       -1.0999788916
C                 2.7965377869       -1.0499873816       -1.2825655690
C                 1.4858870125       -1.2107522506       -0.8097999817
N                 0.8188630536       -2.4289961625       -0.9346499022
C                 0.8357670795       -0.1356300458       -0.1491516497
N                -0.4998147183       -0.3873706569        0.2952928556
C                -0.6870577571       -0.9126438042        1.6612777752
C                -2.1874024413       -0.6822413210        1.9650912935
C                -2.7986842558       -0.2794754327        0.6105610156
C                -1.6463002957        0.4429821707       -0.1024882958
C                -1.7953856765        0.5355625446       -1.6187502522
H                 0.3346538483        1.9058578489        1.6742366982
H                 0.2061147766        2.8048339478        0.1631171592
H                 1.6719106397        2.9327250193        1.1371681479
H                 3.3627887394        2.1484844705       -0.2793635535
H                 4.4734034791        0.2777782148       -1.4713992887
H                 3.2870746476       -1.8779846582       -1.7896802127
H                 1.0946924759       -2.9865523785       -1.7335782243
H                -0.1867669735       -2.3126872246       -0.8556568842
H                -0.4065538271       -1.9734389508        1.6928667448
H                -0.0501872977       -0.3936709249        2.3940323395
H                -2.3030432067        0.1353126993        2.6857950637
H                -2.6664181060       -1.5651842336        2.3994427243
H                -3.6883294467        0.3510453328        0.7128463889
H                -3.0799244660       -1.1691254442        0.0325572836
H                -1.5839176119        1.4701491364        0.2999276211
H                -0.9148087807        0.9990663069       -2.0764085012
H                -2.6706974227        1.1411836308       -1.8807592468
H                -1.9198710861       -0.4622414296       -2.0564013433


