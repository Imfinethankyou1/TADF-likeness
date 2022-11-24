%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/46815394.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.7488892251       -3.8345475602       -2.4606373270
O                -1.7093483144       -2.4269333085       -2.4518639582
C                -2.0403211218       -1.7696965555       -1.3173131952
C                -2.5505813419       -2.3766332708       -0.1702401670
C                -2.8652356748       -1.6174878468        0.9367288877
C                -2.7104566125       -0.2315476547        0.9349919654
C                -3.0448601546        0.5182318629        2.1666035008
C                -2.9945762777        2.0234621804        2.0841562245
O                -3.3359623522       -0.0533578889        3.1966324078
C                -2.2101796358        0.3726341392       -0.2171466448
C                -1.8612830987       -0.3762388688       -1.3234123304
C                -1.1800823278        0.3030061586       -2.4835984546
N                 0.1023267778        0.8230669169       -2.0408750820
C                 0.3272250251        2.1896306465       -1.8141121574
O                -0.4495131279        3.0722728832       -2.0577236375
N                 1.5765100624        2.2898743291       -1.3016476099
C                 2.2380594739        1.0174932317       -1.1331246534
C                 3.4563340086        0.9060642341       -2.0513251210
C                 2.5249377615        0.6921287161        0.3190076988
C                 3.7264032430        0.1358488077        0.7376444634
C                 3.9114255635       -0.1863332392        2.0705583404
F                 5.0920549496       -0.7143868330        2.4407789442
C                 2.9337662601        0.0142841060        3.0270422005
C                 1.7306585401        0.5615889801        2.6225816003
C                 1.5452914992        0.8877483649        1.2923183809
F                 0.3555777305        1.4233898199        0.9457760062
C                 1.1401511478        0.0483515300       -1.6369387097
O                 1.2174598449       -1.1507344188       -1.6727705938
H                -2.7681380091       -4.2092779152       -2.3226220916
H                -1.3870879150       -4.1305088340       -3.4439455239
H                -1.0934495981       -4.2548908635       -1.6928796540
H                -2.6912522167       -3.4450624719       -0.1314797271
H                -3.2431422785       -2.0887653342        1.8318512846
H                -3.6534889771        2.3795123708        1.2959800075
H                -3.3036605000        2.4467955710        3.0365655605
H                -1.9810236591        2.3464790385        1.8577838928
H                -2.0578971294        1.4394353413       -0.2595852648
H                -1.0136732785       -0.4088802139       -3.2932200203
H                -1.7525487468        1.1635526209       -2.8383829901
H                 1.9734570885        3.1681476044       -1.0168416630
H                 3.1787242237        1.2408278217       -3.0486470610
H                 4.2715366962        1.5278900975       -1.6889625783
H                 3.7778152626       -0.1298549007       -2.1146784860
H                 4.5270850445       -0.0608464785        0.0456963150
H                 3.1126111199       -0.2565447814        4.0544810294
H                 0.9272432501        0.7308138652        3.3214199912


