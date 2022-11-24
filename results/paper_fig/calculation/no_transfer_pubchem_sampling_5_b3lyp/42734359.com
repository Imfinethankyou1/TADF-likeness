%chk=./calculation/no_transfer_pubchem_sampling_5_b3lyp/42734359.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.5904086859       -3.4812856268        2.7201824436
O                 6.0766076080       -2.5662604527        1.7737896952
C                 5.1719759130       -1.8136263029        1.0803979180
C                 3.7983429454       -1.8767973582        1.2285282798
C                 2.9685171069       -1.0644598688        0.4674099648
C                 3.4941573504       -0.1829760072       -0.4564553003
C                 2.5988508769        0.7194105092       -1.2533349920
C                 2.4414948438        2.0755604602       -0.5533753397
N                 1.4384869192        2.8686371772       -1.2237348872
C                 1.6238915091        4.1531867239       -1.5595419554
O                 2.6528710614        4.7635552756       -1.3301578461
C                 0.4749022021        4.8460272447       -2.2967613431
C                -0.9055128167        4.1969712692       -2.2392987412
C                -1.4503849724        4.1685844229       -0.8186118852
S                -2.9617885271        3.1544978483       -0.6725973880
C                -2.2461170900        1.5803378265       -0.7921333802
N                -0.9971627036        1.3273800366       -1.0882879105
N                -0.8246124320        0.0002471569       -1.0460402747
C                -1.9610053155       -0.5601645423       -0.7277129642
C                -2.1588841734       -1.9944105619       -0.6323950777
C                -1.0683982569       -2.7905265265       -0.2830534464
C                -1.2098821899       -4.1610978700       -0.1922700237
C                -2.4357590544       -4.7526176185       -0.4530843843
C                -3.5155808682       -3.9682081141       -0.8234919873
C                -3.3829888745       -2.5954296296       -0.9183844015
N                -2.9205199190        0.4152473322       -0.5495672221
C                -4.2438665944        0.2902622261       -0.0832946019
C                -5.2824350138        0.9068870809       -0.7669300040
C                -6.5720847720        0.7757141828       -0.2844880734
C                -6.8215531101        0.0362282677        0.8575896436
C                -5.7846740824       -0.5817736225        1.5477852780
C                -6.0463787048       -1.3636010871        2.7970291233
C                -4.4897180941       -0.4416545084        1.0710053291
C                 4.8773463275       -0.1114180492       -0.6054276981
C                 5.7215255789       -0.9090629936        0.1451734329
O                 7.0826669738       -0.9050877754        0.0635566355
C                 7.6915341425       -0.0104614956       -0.8323341091
H                 4.9523265186       -4.2425278988        2.2573167093
H                 5.0328730118       -2.9802071134        3.5194281261
H                 6.4708368939       -3.9616566887        3.1445188959
H                 3.3559822171       -2.5580168933        1.9383255489
H                 1.8986739769       -1.1241736705        0.6002927736
H                 3.0089350620        0.8793671629       -2.2518326041
H                 1.6090757284        0.2687735296       -1.3487166083
H                 2.1440234960        1.9066266861        0.4893046725
H                 3.3764421215        2.6397948235       -0.5601395539
H                 0.5606479370        2.3811191845       -1.3801735637
H                 0.4283138409        5.8658067336       -1.9076832433
H                 0.7880074451        4.9212357924       -3.3415206953
H                -1.5910899261        4.7677806099       -2.8709575148
H                -0.8590615418        3.1834186284       -2.6394477260
H                -0.7269050762        3.7600340440       -0.1155526895
H                -1.7516523868        5.1631181049       -0.4901008364
H                -0.1188791958       -2.3162773575       -0.0915212456
H                -0.3618760174       -4.7705455919        0.0817891751
H                -2.5475469488       -5.8239304404       -0.3774985357
H                -4.4675464349       -4.4273270196       -1.0454382982
H                -4.2259723766       -1.9972691819       -1.2309726850
H                -5.0806608694        1.4728132504       -1.6629604118
H                -7.3870108678        1.2503310087       -0.8091641014
H                -7.8337445839       -0.0633335411        1.2205431595
H                -5.3542858984       -2.1981641644        2.8830024155
H                -5.9121021452       -0.7236422371        3.6694887214
H                -7.0648234279       -1.7432319593        2.8109333605
H                -3.6659672705       -0.9076324906        1.5925031512
H                 5.2810290729        0.5837103214       -1.3244095742
H                 8.7634002865       -0.1664485249       -0.7198510984
H                 7.4037476250       -0.2176619871       -1.8688238688
H                 7.4505342543        1.0303058501       -0.5911133616


