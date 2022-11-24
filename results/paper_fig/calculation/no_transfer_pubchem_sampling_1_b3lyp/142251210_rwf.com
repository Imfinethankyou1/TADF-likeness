%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/142251210_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/142251210_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -3.2789087607        1.5718558201        0.6068396426
C                -3.1906362628        0.2457319193       -0.1619939865
C                -3.1492679071       -0.9631274615        0.8130332909
C                -1.6643389162       -1.2512235014        1.0435236946
C                -0.9409303478       -2.0553226662       -0.0120267130
C                -0.8462491921       -0.5464197236       -0.0315314239
C                 0.4851767418        0.1706404416        0.1981208609
C                 1.3413426596       -0.3726952007        1.3546140103
C                 2.5520251026        0.5525612402        1.6435108410
C                 3.0130757295        1.3489577027        0.4008457983
C                 2.8356092095        0.5539582341       -0.8995162306
N                 3.7699570315       -0.5865678090       -0.9051489515
C                 1.3281804821        0.1950949299       -1.1053717553
C                -1.8633309342        0.1465332412       -0.9629032179
H                -4.2027748141        1.6285001018        1.1950842436
H                -2.4357398103        1.6799702058        1.3012005967
H                -3.2613879328        2.4302757323       -0.0754806440
H                -4.0562274728        0.1782849753       -0.8336841763
H                -3.6554523892       -1.8362418455        0.3800278188
H                -3.6661114981       -0.7352730168        1.7526019126
H                -1.3037981321       -1.3281310557        2.0657833540
H                -1.5332954962       -2.5507004681       -0.7808586660
H                -0.0690229237       -2.6239892709        0.3031731185
H                 0.2344244145        1.2124436066        0.4553203695
H                 1.6931664254       -1.3821365302        1.0986691753
H                 0.7407336177       -0.4837630351        2.2642949836
H                 3.3832638025       -0.0463696611        2.0343273403
H                 2.2920440683        1.2607885130        2.4397920423
H                 2.4249701770        2.2721906542        0.3185349366
H                 4.0614505055        1.6464516957        0.5081074479
H                 3.1520047288        1.1902509014       -1.7355119548
H                 3.6673282701       -1.1006824771       -1.7799035212
H                 3.5201507169       -1.2468374626       -0.1698674972
H                 1.2668820026       -0.7922000780       -1.5838427657
H                 0.8741496066        0.9037833343       -1.8104724060
H                -2.0063346194       -0.4273346185       -1.8877941039
H                -1.5121298826        1.1407436323       -1.2646794644


