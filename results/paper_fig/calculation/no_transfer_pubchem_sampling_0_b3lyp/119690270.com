%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/119690270.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -2.6235646415        3.1785015158       -0.8305978373
C                -2.0992963828        1.8058002868       -0.4011475363
N                -1.7629315975        0.9619262373       -1.5264273330
C                -0.5383411521        1.3460603405       -2.2237507778
C                 0.6688423208        1.0264183318       -1.3833287563
C                 1.4741473180        2.0118462676       -0.8396417810
C                 2.5517999285        1.6576554985       -0.0499490149
C                 2.8421950357        0.3290002906        0.2383288963
O                 3.9165298650        0.1470004162        1.0587643148
C                 4.6021913151       -1.0980782794        1.0881781463
C                 5.9265623573       -0.8595110963        1.7318545930
C                 7.0604905269       -0.3702450678        1.1601927628
C                 8.1274975391       -0.2341056392        2.0755402348
C                 7.7863107064       -0.6186775444        3.3309874970
S                 6.1547692632       -1.1493261785        3.4223743057
C                 2.0095204546       -0.6770589353       -0.2897373097
O                 2.2809173673       -1.9827879084        0.0341238805
C                 1.4599892568       -2.9911047500       -0.5044023295
C                 0.9473061409       -0.3052916021       -1.0996505750
C                -2.3506343993       -0.2012175393       -1.8949074689
O                -1.9151825610       -0.8464193501       -2.8376359916
C                -3.5436690212       -0.7645326375       -1.1314816970
C                -3.1666865333       -1.3028016357        0.2626555621
C                -4.4800195619       -1.2564715367        1.0368147226
C                -5.1626584524        0.0307546248        0.5625084891
N                -6.6053298624       -0.0703958023        0.7003681653
C                -4.7750296841        0.1398605374       -0.9242042034
H                -1.8425181328        3.7764374719       -1.2915149820
H                -3.4381301752        3.0742721254       -1.5437323323
H                -2.9894524098        3.7154475549        0.0408499307
H                -2.8487177617        1.3162015823        0.2122631170
H                -1.1954463705        1.9402771551        0.2061802103
H                -0.5143052334        0.7731767566       -3.1538218215
H                -0.5748075259        2.4110313231       -2.4585402500
H                 1.2763770883        3.0544149226       -1.0398516758
H                 3.1988206986        2.4085092305        0.3752593185
H                 4.7486293150       -1.4717601271        0.0676644753
H                 4.0285865023       -1.8367189164        1.6588454198
H                 7.1319876768       -0.1184802480        0.1171109120
H                 9.1025106300        0.1310742942        1.8067496219
H                 8.3923083404       -0.6250555257        4.2150466817
H                 0.4220168448       -2.8848119578       -0.1740012915
H                 1.8657465960       -3.9274629064       -0.1237145799
H                 1.4929289775       -2.9988952550       -1.5979422610
H                 0.3015324506       -1.0538451043       -1.5284964254
H                -3.8505923603       -1.6199475780       -1.7407374335
H                -2.7639078744       -2.3130858078        0.2000973159
H                -2.4131226043       -0.6775429203        0.7444054866
H                -5.1147325834       -2.0968308584        0.7548582197
H                -4.3307662998       -1.2764816963        2.1159876652
H                -4.7499909993        0.8895066753        1.1271413067
H                -6.8597203390       -0.2102598854        1.6721949664
H                -7.0539457671        0.7811612373        0.3806242841
H                -4.5972230343        1.1692089273       -1.2264920862
H                -5.6097881954       -0.2363373142       -1.5154537513


