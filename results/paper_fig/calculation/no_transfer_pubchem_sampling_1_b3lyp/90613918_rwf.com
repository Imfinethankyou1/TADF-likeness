%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/90613918_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/90613918_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
O                 2.9471547740        2.9685550194       -1.0142611347
S                 3.0349676377        1.5168175028       -0.8526493287
O                 3.1465700614        0.6284946095       -2.0167808256
N                 1.7312099481        1.0980914497        0.1192622131
C                 1.2883061431       -0.3022383226        0.0910888000
C                 0.0616420131       -0.4739994011        0.9922964180
N                -1.1348620373        0.1499999849        0.4450497101
N                -1.1215449990        1.4834894645        0.1766592868
C                -2.3323105783        1.7496665888       -0.3120504773
C                -3.1436503802        0.5959570371       -0.3593264325
C                -2.3378793280       -0.4163401703        0.1385326374
C                -2.7285624807       -1.8571903313        0.2825184664
C                -4.2617724040       -1.9816975673        0.1402481851
C                -4.8053008452       -1.1410292669       -1.0281615303
C                -4.5591038280        0.3662380175       -0.8121225979
C                 4.4170212317        1.1125067478        0.1860798923
C                 5.4274886691        0.2507244843       -0.1574660936
C                 6.4428850062        0.1836100874        0.8392768897
C                 6.1827624022        0.9983864798        1.9094417130
S                 4.6960803365        1.8680752347        1.7305080323
H                 0.9518860325        1.7482849284       -0.0379032547
H                 2.0975861812       -0.9275179645        0.4853085810
H                 1.0593980879       -0.6472984825       -0.9254375020
H                 0.2689195260       -0.0515716877        1.9818088730
H                -0.1508419822       -1.5392953084        1.1101065551
H                -2.5728460691        2.7617433207       -0.6129203262
H                -2.4093499196       -2.2680055829        1.2494859911
H                -2.2343216013       -2.4651430642       -0.4902501709
H                -4.7359819631       -1.6489114437        1.0735617425
H                -4.5288502818       -3.0364598425        0.0079565009
H                -4.3152720247       -1.4592629564       -1.9588974617
H                -5.8779022535       -1.3288283427       -1.1552946315
H                -5.2675446842        0.7476166496       -0.0614750779
H                -4.7701066626        0.9185386851       -1.7365066064
H                 5.4285932948       -0.2991062729       -1.0908085410
H                 7.3253443630       -0.4417878331        0.7626409131
H                 6.7801896144        1.1388855487        2.8011315918


