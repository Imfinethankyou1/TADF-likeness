%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/82397031_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/82397031_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -3.7329114249        1.3341376194       -0.3617730990
C                -2.5698274277        0.3513552412       -0.5048797593
C                -1.2429116308        1.0003651956       -0.1033670778
N                -0.1227541841        0.0818060432       -0.2850479533
C                 1.1837801201        0.5621982312       -0.2776782095
C                 1.4662440801        1.9163989407        0.0016078401
C                 2.7753583690        2.3958528064       -0.0285029692
C                 3.8408964961        1.5462249109       -0.3160474266
C                 3.5695173297        0.1993407286       -0.5676757789
C                 2.2714068965       -0.3049863417       -0.5618119623
C                 1.9883416270       -1.7658459854       -0.8205742495
C                 0.7783053869       -2.2099191898        0.0061940906
C                -0.4413455447       -1.3532825903       -0.3349716771
C                -1.6101999039       -1.6284705793        0.6246141054
N                -2.8111119179       -0.8258939407        0.3414655551
H                -3.6037144645        2.1996354902       -1.0223428540
H                -3.8139902203        1.6829394499        0.6731637558
H                -4.6800319901        0.8495065767       -0.6262272664
H                -2.4811591419        0.0804754506       -1.5737553032
H                -1.3165500091        1.3534554612        0.9401892673
H                -1.0899134699        1.8850523688       -0.7338007345
H                 0.6660351902        2.6002574231        0.2590337102
H                 2.9556078014        3.4460756717        0.1881806612
H                 4.8616430513        1.9166169344       -0.3332059515
H                 4.3883125446       -0.4867430347       -0.7779081295
H                 2.8689264359       -2.3714349749       -0.5758675533
H                 1.7776633142       -1.9401100093       -1.8871288203
H                 1.0085452928       -2.1123311526        1.0755100818
H                 0.5337177419       -3.2617895699       -0.1832415416
H                -0.7630957908       -1.6232141641       -1.3564044156
H                -1.2607762806       -1.4208342764        1.6465731505
H                -1.8681619156       -2.6935535482        0.5879216558
H                -3.5258453605       -1.4032851861       -0.0922041412


