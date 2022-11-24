%chk=calculation/no_transfer_pubchem_sampling_4_b3lyp/154027315_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_4_b3lyp/154027315_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.5783576436        0.8985884782        0.4549553970
O                 6.1299522555        0.8312857056       -0.8870003658
C                 4.8329076379        0.4548277783       -1.1175706653
C                 4.4449640741        0.4110743286       -2.4627340583
C                 3.1494124440        0.0357156558       -2.7892466878
C                 2.2148966993       -0.3041752625       -1.7990906988
C                 2.5880917864       -0.2524221199       -0.4464711108
C                 3.9055578905        0.1312841156       -0.1238926679
Si                1.3392229197       -0.6733382949        0.9274109123
C                 1.7068731862        0.5672682590        2.3156475739
C                 1.6316618491       -2.4691609557        1.4506955170
C                -0.3436991906       -0.4075745818        0.0480965382
C                -0.3445291323       -0.3541136010       -1.3539316875
C                -1.5208888498        0.0340823959       -2.0187435953
C                -2.7003039302        0.3530826776       -1.3686856325
C                -2.7586051417        0.2194560713        0.0420664958
C                -1.5679185467       -0.1555366229        0.7342596636
C                -1.6227072079       -0.2901713207        2.1983582013
O                -0.7669589212       -0.8789987029        2.8531812886
C                -2.8037990400        0.3752706046        2.8821325059
C                -4.0871669439        0.0936468836        2.1138057523
N                -3.9538590089        0.4521410047        0.7079762997
C                -5.2293521021        0.6158090709       -0.0014432994
C                -5.1718076975        0.1735602402       -1.4612716874
C                -3.9332591699        0.7747970675       -2.1297198923
C                 0.8360336488       -0.7758256489       -2.2157359685
H                 6.5164610714       -0.0782443589        0.9537535178
H                 7.6234161102        1.2117588147        0.4072462659
H                 6.0072249853        1.6337120513        1.0382206777
H                 5.1693145548        0.6779908134       -3.2263736890
H                 2.8559875527        0.0046904231       -3.8370072842
H                 4.2049958708        0.1778973765        0.9174632805
H                 2.7816920002        0.5850666191        2.5319178667
H                 1.1800318273        0.3241887110        3.2392983366
H                 1.4244525729        1.5818859818        2.0092620968
H                 2.6586446670       -2.5985730288        1.8142020026
H                 1.4945905924       -3.1538387645        0.6053810244
H                 0.9401205617       -2.7557009033        2.2471170113
H                -1.5101135848        0.0922944777       -3.1064972227
H                -2.8710067723        0.0182039669        3.9129775980
H                -2.6195155984        1.4591610026        2.9034340455
H                -4.9141150846        0.6728063312        2.5396307146
H                -4.3566751024       -0.9743005279        2.2114405618
H                -5.5437324932        1.6695437608        0.0563202608
H                -5.9903081443        0.0298367473        0.5302901889
H                -5.1212702056       -0.9209540322       -1.5167859250
H                -6.0928375661        0.4863569278       -1.9658126741
H                -4.0246965282        1.8718215816       -2.1379020479
H                -3.8576979185        0.4598443263       -3.1764167359
H                 0.6477630146       -0.4576050744       -3.2478501039
H                 0.8441994647       -1.8784184490       -2.2563598948


