%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/66874476.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
O                 1.1370136825        0.2328775083        2.7062283378
C                 0.6189235692        0.2120903947        1.4034645908
C                 1.8033567118        0.1202396434        0.4527871144
C                 2.8255354428        1.0538599873        0.5621731928
C                 3.9104370925        0.9994932554       -0.2931779891
C                 3.9826240795        0.0143449409       -1.2640561019
C                 2.9720510925       -0.9251455447       -1.3655358364
C                 1.8867028157       -0.8766057811       -0.5075747644
C                -0.2198151975        1.4687302603        1.1140157431
C                -0.7171072366        1.4137266079       -0.3230656884
C                -1.4651930942        0.0956606928       -0.5949524800
C                -2.9587737787        0.3424029349       -0.8327482246
C                -3.7165973715       -0.9477215402       -0.9512088088
C                -3.3334414328       -2.0097813821       -0.2449637480
C                -2.1763783768       -1.9408412673        0.6161640469
C                -1.3017408564       -0.9297288971        0.5068113837
O                -0.2170602096       -0.9283896048        1.3338846571
H                 0.4493684681       -0.0733893882        3.3095827474
H                 2.7737497448        1.8089593952        1.3309973894
H                 4.7028312619        1.7276780662       -0.2013072679
H                 4.8287552709       -0.0253900811       -1.9342426079
H                 3.0280889379       -1.7032451508       -2.1126748645
H                 1.1090195267       -1.6220147423       -0.5796238544
H                 0.3923757612        2.3533705053        1.2840559356
H                -1.0617099729        1.4859685043        1.8123430909
H                 0.1405194634        1.5048270611       -0.9926432053
H                -1.3793725525        2.2600325468       -0.5139476602
H                -1.0536940279       -0.3639856198       -1.5055794096
H                -3.3728891408        0.9147301552        0.0082120511
H                -3.0838632088        0.9448170289       -1.7338456396
H                -4.6018823622       -0.9576717035       -1.5675880644
H                -3.8845263571       -2.9375090481       -0.2740812264
H                -2.0173057453       -2.7123887380        1.3521001610


