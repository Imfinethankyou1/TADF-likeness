%chk=./calculation/no_transfer_pubchem_sampling_5_b3lyp/160299960.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 4.1031642987        0.3017437129       -3.1301734719
C                 5.0113397513        0.2225106267       -1.9406070643
C                 6.3822643061        0.0971740672       -2.1418111944
C                 7.2632820452       -0.0067469106       -1.0857237576
C                 6.7757662357       -0.0011099644        0.2200708403
C                 7.6675768928       -0.1239252138        1.3519343391
N                 7.3219364008       -0.1478548612        2.6093481341
C                 8.4854250905       -0.2766184446        3.3011343633
C                 9.5159110185       -0.3270944999        2.4160103337
O                 9.0017993214       -0.2304343123        1.1724386997
C                 5.4061805830        0.1221900145        0.4349518943
C                 4.5227518017        0.2466270971       -0.6283774962
C                 3.0839554093        0.3864192001       -0.3299732242
C                 2.4467652241       -0.5575405294        0.4686968937
C                 1.1016710266       -0.4369845189        0.7615958783
C                 0.3673949147        0.6394656751        0.2831775087
C                -1.1041522893        0.7490207498        0.5615542645
C                -1.9185814332        0.3885913185       -0.6713386897
O                -1.4956828349        0.5223957883       -1.7913482266
C                -3.3025300494       -0.1954992312       -0.4054258008
N                -4.0159928695       -0.3410113419       -1.6569547356
C                -4.1394748140        0.6851333735        0.5544325408
C                -5.3677705572       -0.0332888561        1.0043315105
C                -5.5215767307       -0.7342164367        2.1650593539
N                -6.7777988986       -1.2765290397        2.2290792947
C                -7.4650929534       -0.9325881161        1.1007836617
C                -8.7617191795       -1.2364038364        0.7049306720
C                -9.1875767523       -0.7444452292       -0.5095823407
C                -8.3469225900        0.0295567130       -1.3157306579
C                -7.0616533565        0.3344225211       -0.9299005025
C                -6.6040503588       -0.1488584747        0.2958733714
C                 1.0058251248        1.5921724472       -0.5002712431
C                 2.3468049809        1.4659428332       -0.8066513615
H                 4.0391661838        1.3285792145       -3.4907405112
H                 4.4935116719       -0.3087260519       -3.9421841120
H                 3.0990853719       -0.0341755018       -2.8845918011
H                 6.7617956814        0.0765590610       -3.1536760274
H                 8.3240939692       -0.1006010166       -1.2616111997
H                 8.4975398546       -0.3230696784        4.3701748808
H                10.5767855482       -0.4219311731        2.5181516891
H                 5.0421267869        0.1340859764        1.4504132141
H                 3.0096191995       -1.3991350933        0.8453950568
H                 0.6180916078       -1.1872966167        1.3718436146
H                -1.3907328383        0.1098552003        1.3972153524
H                -1.3591193739        1.7835951061        0.8091770405
H                -3.1158848638       -1.1626850273        0.1002750772
H                -3.3576435044       -0.3697982804       -2.4290826808
H                -4.5966611128       -1.1680687386       -1.6745009968
H                -4.4180669414        1.5890987083        0.0126543163
H                -3.5479845217        0.9564524731        1.4286859716
H                -4.8226185282       -0.8824082692        2.9670259315
H                -7.1404584654       -1.8208964090        2.9925477966
H                -9.4096034715       -1.8356681383        1.3283081954
H               -10.1901164014       -0.9605325623       -0.8487623176
H                -8.7221423096        0.3918892647       -2.2616611177
H                -6.4091374674        0.9181762055       -1.5595616240
H                 0.4450806645        2.4338569384       -0.8780424351
H                 2.8340375013        2.2206260873       -1.4054711011


