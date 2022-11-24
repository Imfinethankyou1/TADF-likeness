%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/23109275.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 0.4371304329        2.6624068799       -1.2142452261
C                 0.0801495684        1.3327321503       -0.6228534243
C                -1.2528918902        1.0032496328       -0.4492068748
C                -1.6386271029       -0.2322584320        0.0871523102
C                -3.0428430657       -0.6047259552        0.2910001337
C                -4.0788416242       -0.0308557678       -0.4433224520
C                -5.3894564478       -0.4090261969       -0.2174521254
C                -5.6869888516       -1.3627399851        0.7414992398
C                -4.6627244571       -1.9477901183        1.4687818828
C                -3.3522809622       -1.5751925489        1.2441496040
N                -0.7334636615       -1.1338598252        0.4526480700
C                 0.5321771532       -0.8298154970        0.2900896676
S                 1.7923634247       -1.9486819081        0.7120467400
C                 2.9905280023       -0.7847314906        0.1849924145
O                 4.2904643698       -1.2462341484        0.2549327529
C                 5.2920527457       -0.4825264796        0.7410603701
N                 6.4338910052       -1.2003164516        0.7834740535
O                 5.1979069225        0.6789005623        1.0556623074
C                 2.4802425481        0.3947333810       -0.2781774167
N                 3.2377661223        1.4241477991       -0.8101645524
C                 1.0394538963        0.3817106755       -0.2332794484
H                 1.1157867393        2.5386941763       -2.0560570005
H                -0.4604865354        3.1674662182       -1.5641941034
H                 0.9087826060        3.3021862001       -0.4675456471
H                -2.0082159652        1.7245781017       -0.7193784729
H                -3.8648457097        0.6966849428       -1.2119888153
H                -6.1835766329        0.0406339709       -0.7960229612
H                -6.7126672771       -1.6527241421        0.9165153362
H                -4.8892019298       -2.6972901888        2.2130384942
H                -2.5437621328       -2.0243468456        1.7990914613
H                 7.2626734488       -0.7763249771        1.1580819305
H                 6.4233071138       -2.1788666993        0.5597282601
H                 2.8030456503        2.3293315026       -0.7241486714
H                 4.1831534963        1.4408534640       -0.4439288367


