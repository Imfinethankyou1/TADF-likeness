%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/46406518.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 8.2005836187       -2.1185333729        0.5511196777
C                 7.0283279567       -1.6018200550       -0.2255542761
C                 6.8819140330       -1.8875318066       -1.5754918162
C                 5.7818225335       -1.4218877146       -2.2734601565
C                 4.8005699000       -0.6623735718       -1.6572791528
C                 3.6143984418       -0.1709889942       -2.4381291106
C                 4.9498633830       -0.3708011848       -0.2939568726
N                 3.9666492859        0.4051867220        0.3282096060
C                 3.8047011919        0.7264156500        1.6320141101
O                 4.5363370022        0.4301536400        2.5503423564
C                 2.5528265401        1.5900036706        1.8363833494
N                 1.5259832676        1.2650724733        0.8667751390
C                 0.8987731487       -0.0201256061        1.1661475052
C                -0.4883974509       -0.0852846826        0.5348911011
N                -1.3984442722        0.8941521649        1.1187396747
C                -2.1841780920        0.3397534564        2.2059272679
C                -3.3394610491       -0.5207130551        1.6781148125
O                -3.8081899270       -1.4321189926        2.3242931460
N                -3.7455076375       -0.0935499498        0.4613151538
C                -4.7536661012       -0.5774083594       -0.3768294294
C                -5.5871177310       -1.6175299775        0.0120402743
C                -6.5871915629       -2.0860425927       -0.8288166731
C                -7.4801372990       -3.2004591490       -0.3757782706
C                -6.7441872146       -1.4941003515       -2.0729204431
C                -5.9150754069       -0.4565713115       -2.4633390193
C                -4.9125352850        0.0205411102       -1.6357855001
C                -4.0252531723        1.1476881606       -2.0833947125
C                -0.7369258244        2.1558161245        1.4626533510
C                -1.6927923667        3.3161142634        1.1869842227
C                 0.5394962629        2.3106090253        0.6296661520
C                 6.0546321490       -0.8408529727        0.4046208684
H                 9.0654905402       -2.2539726897       -0.0942143201
H                 8.4600615559       -1.4363126953        1.3567595268
H                 7.9550785428       -3.0838709416        0.9947965358
H                 7.6298961810       -2.4765773323       -2.0857461924
H                 5.6830969927       -1.6543574723       -3.3244505405
H                 2.6812148924       -0.5412775145       -2.0115178138
H                 3.5783256372        0.9192025579       -2.4543392732
H                 3.6758482353       -0.5197603927       -3.4658631080
H                 3.2113340833        0.7304544406       -0.2645322657
H                 2.8645278270        2.6296062484        1.6970326971
H                 2.2280434933        1.4703690375        2.8822780619
H                 0.8007995899       -0.1696578607        2.2539186475
H                 1.5332537386       -0.8248048947        0.7818264769
H                -0.4068805523        0.1006083394       -0.5411367126
H                -0.9131160795       -1.0810991944        0.6706669380
H                -1.5955637733       -0.2670498080        2.9094506509
H                -2.6375013566        1.1554765338        2.7761469329
H                -3.1755658906        0.6613307758        0.0942992749
H                -5.4470751429       -2.0635328464        0.9829713799
H                -6.8948905444       -4.0001118991        0.0733845056
H                -8.0526974671       -3.6044528666       -1.2070350886
H                -8.1780488572       -2.8366402487        0.3781706794
H                -7.5161632085       -1.8426935065       -2.7432850552
H                -6.0508797433       -0.0073512025       -3.4371009207
H                -2.9782847931        0.8418840508       -2.0981495206
H                -4.1248381332        2.0103268343       -1.4231777103
H                -4.2989319084        1.4601683997       -3.0879770273
H                -0.4539139231        2.1704701068        2.5291716512
H                -1.9176152697        3.3663797155        0.1240978814
H                -2.6250029066        3.1852190003        1.7304395517
H                -1.2448494188        4.2574084630        1.4944486246
H                 0.9912284350        3.2812571244        0.8393692904
H                 0.2696220061        2.2951016478       -0.4327344559
H                 6.1461828956       -0.6074026707        1.4525283629


