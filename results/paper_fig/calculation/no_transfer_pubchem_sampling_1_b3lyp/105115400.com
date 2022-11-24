%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/105115400.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -0.9908362400        4.1960379881        0.7631740532
C                -0.8414004092        2.7774457905        1.2960470788
N                -0.3165402872        1.9132554840        0.2544247994
C                -0.0934681611        0.5204060689        0.6605252751
C                -1.1150230655       -0.4356465907        0.0130806540
C                -2.5783654740       -0.1036677732        0.3178301235
C                -2.9483574300       -0.3983268079        1.7691462503
C                -3.4913182830       -0.9061346848       -0.6146844601
O                -3.2864653552       -2.2867467091       -0.4140917090
C                -4.0896348904       -3.0824292744       -1.2390452122
C                 1.3103021904        0.0575455547        0.3337668668
C                 2.0817079931       -0.4722441100        1.3593715750
C                 3.3767389427       -0.9187531888        1.1666540331
C                 3.9372382823       -0.8354155765       -0.1003552498
O                 5.2008251984       -1.2317638455       -0.4322320362
C                 6.0113942011       -1.7833635302        0.5746779839
C                 3.1726833096       -0.3065760553       -1.1356568851
C                 1.8737283324        0.1378711975       -0.9455636937
C                 1.1492196871        0.6667626034       -2.1522600069
H                -1.4417433614        4.8370262769        1.5163172675
H                -0.0197273732        4.6067446293        0.4938463204
H                -1.6210097300        4.1937231617       -0.1221820278
H                -0.1962733200        2.7889914629        2.1948081049
H                -1.8206369108        2.3978135026        1.5981151769
H                 0.5620952115        2.3100596573       -0.0652780936
H                -0.2129849187        0.4460893247        1.7514444381
H                -0.9779852686       -0.4180751767       -1.0661011574
H                -0.9029769448       -1.4484178894        0.3584786843
H                -2.7454277072        0.9545491329        0.0967681622
H                -2.8392141673       -1.4622478816        1.9614649018
H                -2.3100405576        0.1498135257        2.4565518810
H                -3.9809828356       -0.1147980783        1.9629663785
H                -4.5462446361       -0.6542727810       -0.4148911553
H                -3.2789125554       -0.6468553945       -1.6648504677
H                -3.8529527425       -4.1186377140       -1.0049830986
H                -3.8843838336       -2.8926944398       -2.3011892710
H                -5.1571723459       -2.9014574975       -1.0542187053
H                 1.6574861876       -0.5419451675        2.3514783584
H                 3.9263629976       -1.3208391100        2.0028840273
H                 6.1955423423       -1.0664811112        1.3819985120
H                 5.5706256361       -2.6945417595        0.9927611758
H                 6.9559889481       -2.0303801461        0.0922657504
H                 3.6230680884       -0.2513343211       -2.1152623382
H                 1.8058115932        1.3274178293       -2.7167009096
H                 0.8820997224       -0.1643077586       -2.8057109259
H                 0.2471589400        1.2067981827       -1.8882224290


