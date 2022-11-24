%chk=calculation/no_transfer_pubchem_sampling_6_b3lyp/58323961_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_6_b3lyp/58323961_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -9.1847243991        1.0918745029        2.5712528792
C                -8.4851908282        0.3775604824        1.4598098234
C                -8.9891448621       -0.3667585842        0.4005302418
C                -7.8963141370       -0.8257935880       -0.3664673748
C                -6.7264887407       -0.3556617876        0.2336593096
C                -5.3371329489       -0.5297873799       -0.1361059459
C                -4.9455482095       -1.2629946723       -1.2519425798
C                -3.6031348270       -1.4510963921       -1.6387125718
C                -2.5150088213       -0.9250181769       -0.9489459220
C                -1.1485215456       -1.1401130076       -1.3777863903
C                -0.6587459514       -1.8583008464       -2.4804350162
C                 0.7374440948       -1.7713429907       -2.4654578272
C                 1.1106840660       -0.9997078888       -1.3535774170
C                 2.4300237327       -0.6170809902       -0.8939507639
C                 3.5880793678       -1.0082461061       -1.5543533621
C                 4.8879966913       -0.6548754370       -1.1324702848
C                 5.1605309289        0.1314766572       -0.0176846462
C                 6.5026228108        0.5008367415        0.3837888860
C                 7.0036860865        1.0613933246        1.5567444339
C                 8.4068563493        1.1788546755        1.4197939369
C                 8.7584360475        0.7007036740        0.1681747401
C                10.0875992938        0.6080853452       -0.5090895698
N                 7.6006738134        0.2828712337       -0.4406880775
C                 3.9934790638        0.5595825935        0.7130264561
N                 3.9970225386        1.3124707991        1.8171192008
S                 2.4324447186        1.5274046244        2.2661482859
N                 1.6901437282        0.6735520084        1.0715891077
C                 2.6511693856        0.1893553233        0.2787356737
N                -0.0529534530       -0.6344186212       -0.7186054339
C                -2.8611214968       -0.1557839231        0.2166153398
N                -1.9908944011        0.4451070642        1.0347655642
S                -2.8552778488        1.1946321687        2.2130642259
N                -4.3721778486        0.7817802683        1.7252979322
C                -4.2496633593        0.0391974094        0.6172912604
N                -7.1220419995        0.3728067036        1.3423333583
H                -8.9764603200        2.1703155556        2.5643932828
H               -10.2664515429        0.9631846595        2.4758742220
H                -8.8901629430        0.7090653361        3.5577060330
H               -10.0381120153       -0.5525375799        0.2109677660
H                -7.9504576092       -1.4331573270       -1.2595592384
H                -5.7130022571       -1.7215033667       -1.8678202538
H                -3.4190532478       -2.0434972146       -2.5294976695
H                -1.2577876745       -2.3843528295       -3.2106125754
H                 1.4122829626       -2.2181013064       -3.1822916877
H                 3.5014279892       -1.6281947132       -2.4412122569
H                 5.7127667440       -1.0531517823       -1.7173993438
H                 6.4050237444        1.3371313227        2.4104512389
H                 9.0981510637        1.5702151730        2.1544417435
H                10.8707559192        0.9826633990        0.1555906819
H                10.3450915724       -0.4259922010       -0.7756619788
H                10.1255665134        1.2031233513       -1.4320537063
H                 7.5460971186       -0.0272326771       -1.3985748035
H                -0.0940333283       -0.0673624195        0.1241685054
H                -6.4564537290        0.8253404119        1.9576205679


