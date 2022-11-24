%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/49811704.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 7.6332445666       -2.9163351386       -1.0890861601
C                 6.1850880949       -2.4880409298       -1.2261120247
O                 5.3359498180       -3.1785556199       -1.7391973915
N                 5.9665220868       -1.2448658763       -0.7045733087
C                 4.7791650679       -0.5088632082       -0.6420016817
C                 4.8198292768        0.7456760201       -0.0365152196
C                 3.6870266725        1.5281495523        0.0581732474
C                 2.4816502016        1.0628039709       -0.4579767246
N                 1.2930580703        1.7997674293       -0.4064173089
C                 1.0631758934        2.9938431826        0.1771759473
O                 1.8644564851        3.6832898665        0.7705577312
C                -0.4121434061        3.4143852046        0.0305228769
N                -1.2781305910        2.5806070898        0.8436244802
C                -1.8005631296        1.4398320382        0.2710937138
O                -1.4525846969        1.0922975580       -0.8448008995
N                -2.7094347780        0.7289541380        1.0136974786
C                -3.2904450352       -0.4213453274        0.3773321376
C                -4.1317185939       -0.2480743229       -0.7066255182
C                -4.6975307626       -1.3561837746       -1.3232334046
C                -5.6166701141       -1.1744805127       -2.4922084640
C                -4.4116238872       -2.6190641136       -0.8245706709
C                -3.5721048695       -2.7990201906        0.2663403990
C                -3.2930755471       -4.1677519104        0.8063940665
C                -3.0040069391       -1.6834366454        0.8661927896
C                -3.1140858663        0.9938148208        2.3338022971
O                -3.9478785331        0.3162357940        2.8873799318
C                -2.4132754957        2.1217566033        2.8879800693
S                -2.5569510232        2.6849761616        4.5156920821
C                -1.3966048977        3.9101749817        4.2058220101
C                -0.9213235463        3.9080899321        2.9315462140
C                -1.5111367165        2.8778381030        2.1626486960
C                 2.4384096997       -0.1915481922       -1.0628931889
C                 3.5710824127       -0.9733574759       -1.1564921722
H                 8.2823862730       -2.2222908126       -1.6191768733
H                 7.9228412402       -2.9382049913       -0.0404339812
H                 7.7515073556       -3.9091373959       -1.5127371463
H                 6.7658020312       -0.7763276508       -0.3000717155
H                 5.7526476039        1.1158156175        0.3658090064
H                 3.7317151525        2.4959123774        0.5284558481
H                 0.4917346988        1.3488594290       -0.8471306072
H                -0.5231699148        4.4538195277        0.3406697240
H                -0.7242401658        3.3032143038       -1.0113692530
H                -4.3395625931        0.7456202212       -1.0740814139
H                -5.7740069634       -2.1147745306       -3.0140171181
H                -6.5841479270       -0.8053643364       -2.1515954967
H                -5.2082523952       -0.4470841193       -3.1906061324
H                -4.8529626476       -3.4850624734       -1.2976313993
H                -2.2608621509       -4.2494080835        1.1391121233
H                -3.9376458108       -4.3629577318        1.6636048527
H                -3.4847166295       -4.9300218253        0.0554355896
H                -2.3472068417       -1.7954484273        1.7155599872
H                -1.1160214463        4.5769446782        4.9970087460
H                -0.1645476653        4.5809662212        2.5742663071
H                 1.5029548305       -0.5550785523       -1.4632570839
H                 3.5283900483       -1.9415606540       -1.6254249941


