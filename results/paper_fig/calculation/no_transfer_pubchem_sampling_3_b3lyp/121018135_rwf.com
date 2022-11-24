%chk=calculation/no_transfer_pubchem_sampling_3_b3lyp/121018135_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_3_b3lyp/121018135_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.9792192108        0.6859040800       -2.4337854188
O                 6.6480721922        0.7605251442       -1.0532145032
C                 5.4610759709        0.2452559483       -0.6532303186
C                 4.5122117676       -0.3689097172       -1.4771784126
C                 3.3192798213       -0.8564166834       -0.9368729028
C                 3.0478160815       -0.7425227681        0.4296463061
C                 1.7973983370       -1.2466696696        1.0820109298
O                 1.7404190555       -1.4643761020        2.2911135508
N                 0.7228184726       -1.4592337199        0.2518150516
C                -0.5623590705       -1.9148446668        0.7714331192
C                -1.5269623508       -0.7863387604        1.0906592680
C                -1.2603755557        0.1175886684        2.1258191142
C                -2.1868377772        1.1280013744        2.3751200439
N                -3.3150401191        1.2965729069        1.6793537493
C                -3.5771698664        0.4396504150        0.6768463339
C                -4.8450458266        0.6672892995       -0.0700871228
C                -5.1176998406        0.0733541389       -1.3106641591
C                -6.3332520330        0.3362450051       -1.9406504446
N                -7.2819003448        1.1343536649       -1.4368028308
C                -7.0134437757        1.7057933206       -0.2542278723
C                -5.8322738610        1.5121536717        0.4572527908
C                -2.7057671303       -0.6170538182        0.3648801527
C                 4.0078268069       -0.1444050459        1.2635394246
C                 5.1793776551        0.3401429062        0.7241639009
F                 6.0956499383        0.9235370844        1.5172774885
H                 6.2509875348        1.2286331240       -3.0499142063
H                 7.0416058069       -0.3553589929       -2.7748030670
H                 7.9581053098        1.1578165691       -2.5284414782
H                 4.7011346636       -0.4803567765       -2.5385187740
H                 2.6229687994       -1.3628807602       -1.5988518807
H                 0.7275475285       -1.0162330602       -0.6561900959
H                -0.3390898152       -2.4845788184        1.6767788368
H                -1.0117833721       -2.5959849349        0.0403113535
H                -0.3596716788        0.0206894432        2.7237512189
H                -2.0153848024        1.8406489315        3.1803023033
H                -4.3967821150       -0.5717500500       -1.8028139676
H                -6.5533923281       -0.1181420409       -2.9056838593
H                -7.7893435899        2.3558668743        0.1471490309
H                -5.6637477152        2.0032482031        1.4084434579
H                -2.9528521473       -1.3188824961       -0.4260748153
H                 3.8266601628       -0.0683328920        2.3296147042


