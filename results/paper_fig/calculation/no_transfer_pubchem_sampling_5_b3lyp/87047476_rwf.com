%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/87047476_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/87047476_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -4.1325170247       -3.2322606504        3.7707856925
O                -4.6091802828       -3.3833974681        2.4374963036
C                -4.0042522668       -2.6619560054        1.4546073248
C                -3.0308901400       -1.6869654242        1.6787642573
C                -2.4548067429       -0.9959107161        0.5994941658
C                -1.4258983123        0.0490867734        0.8241585319
C                -1.4712131354        1.2391485733        0.0899955350
C                -0.5122726482        2.2418260707        0.2779236476
C                -0.5403801565        3.5206204560       -0.4716825982
C                 0.6238772587        4.3038000915       -0.5668669952
C                 0.6119871797        5.5007254825       -1.2790865863
C                -0.5584721869        5.9436968868       -1.8985985497
C                -1.7236189633        5.1805249335       -1.7992450355
C                -1.7150601391        3.9800238291       -1.0932349761
N                 0.4949675266        2.0795944555        1.1642439797
C                 0.5660017164        0.9650883430        1.8683076617
S                 1.9117340912        0.7954924910        3.0440720591
C                 2.8471100460        2.3290790015        2.6952758867
C                 3.6257551179        2.4165675683        1.3724700885
O                 3.8696959381        3.5193930092        0.9007515166
N                 4.0403057692        1.2207629037        0.8528850200
C                 4.8039561419        0.9769422316       -0.3088058210
C                 5.4550962244        1.9902253789       -0.9916476040
C                 6.2337294868        1.6954329605       -2.1331009088
C                 6.3753336499        0.4075940379       -2.5915689144
C                 5.7170406351       -0.6554893563       -1.9226202754
N                 5.8844721458       -1.9161855879       -2.4206007020
C                 5.2647669368       -2.9103966955       -1.8198391572
C                 4.4248811853       -2.7497129949       -0.6930325575
C                 4.2470712328       -1.4878276237       -0.1752761053
C                 4.9050492070       -0.3799083911       -0.7712900779
C                -0.3643258230       -0.0988181044        1.7496600300
C                -0.1663246093       -1.2856392797        2.5192393818
N                 0.0356297984       -2.2336818442        3.1655225088
C                -2.8705902818       -1.2880518388       -0.7044792449
C                -3.8546014507       -2.2529917829       -0.9342493314
O                -4.2690575019       -2.4618567701       -2.2243148038
C                -3.9771731661       -3.7514049540       -2.7771118450
C                -4.4352403059       -2.9437960138        0.1392606079
O                -5.3689118924       -3.9167162603       -0.1037361176
C                -6.7178644644       -3.5645518862        0.2272127090
H                -3.0576706470       -3.4395017938        3.8356187585
H                -4.6839813766       -3.9623580911        4.3652795250
H                -4.3335765815       -2.2246216355        4.1573695797
H                -2.7278061401       -1.4520501648        2.6898408926
H                -2.2683059007        1.3734495239       -0.6300541626
H                 1.5376785063        3.9683070163       -0.0892221111
H                 1.5232845959        6.0879995586       -1.3492786454
H                -0.5649163652        6.8793533554       -2.4512371200
H                -2.6431454743        5.5230245114       -2.2658657753
H                -2.6383407212        3.4156390903       -1.0053695719
H                 3.5565847201        2.4114269841        3.5270554939
H                 2.1745668584        3.1854503367        2.7273535039
H                 3.7326280527        0.4086689086        1.3715160178
H                 5.3590321091        3.0081240337       -0.6418981907
H                 6.7325870686        2.5140128154       -2.6445769149
H                 6.9765001025        0.1620717353       -3.4605085803
H                 5.4214061178       -3.9035274340       -2.2403283325
H                 3.9282317577       -3.6095447527       -0.2541666845
H                 3.5839266159       -1.3644143710        0.6768541762
H                -2.4304718193       -0.7931233964       -1.5633473616
H                -4.4747409227       -4.5465717149       -2.2157199607
H                -4.3532717752       -3.7281042352       -3.8022942702
H                -2.8938758998       -3.9288414220       -2.7923282749
H                -7.3291135153       -4.4272583968       -0.0462526465
H                -7.0419680491       -2.6891227336       -0.3487856717
H                -6.8210511110       -3.3665965576        1.2986066268


