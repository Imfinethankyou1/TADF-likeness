%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/24055153.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.6509926878       -1.3396230395       -3.6005757856
O                 5.4857093346       -0.5553275307       -3.5117044868
C                 4.6345510454       -0.7777285514       -2.4714325979
C                 4.8113225708       -1.7296000400       -1.4791446573
C                 3.8669311151       -1.8599348212       -0.4721869991
C                 2.7463442885       -1.0506439432       -0.4377727460
N                 1.7543469657       -1.1362660056        0.5460441454
C                 1.7682001828       -1.8219931852        1.7049072837
O                 2.6521118589       -2.5521528565        2.0986586268
C                 0.4855446066       -1.6387442111        2.5370740784
C                -0.2998943272       -2.9476528377        2.5372979755
O                -0.2779306539       -0.5619051240        1.9831964404
C                -1.2452628564       -0.0042801425        2.7156548599
O                -1.5343164887       -0.3258182400        3.8365687383
C                -1.8899067004        1.0860727702        1.9094707867
C                -2.3446160934        0.8032633287        0.6555963543
C                -2.3000008456       -0.5780236587        0.1313716070
C                -1.4907822795       -0.9045480332       -0.9519646547
C                -1.4496084842       -2.2033174325       -1.4255302144
C                -2.2288719920       -3.1844773700       -0.8347623381
C                -3.0439955431       -2.8634890909        0.2381817580
C                -3.0764169927       -1.5680983963        0.7237308814
C                -2.9209926133        1.8607462681       -0.1259052451
C                -3.4346476477        1.6606458608       -1.4153129098
C                -3.9656053750        2.7113479942       -2.1263097015
C                -4.0078528697        3.9911081973       -1.5751767382
C                -3.5215733259        4.2051646138       -0.3063417653
C                -2.9775514265        3.1500425249        0.4308543783
C                -2.4850073989        3.4265265829        1.7786160799
O                -2.5110576524        4.5293905100        2.2953615472
N                -1.9617316879        2.3361009278        2.4627600980
C                -1.4405045435        2.5719010346        3.7942053917
C                 2.5678813514       -0.0768782517       -1.4417369235
O                 1.4386112588        0.6931011350       -1.3309379198
C                 1.2355612825        1.7224454952       -2.2701489221
C                 3.5060788405        0.0476025602       -2.4452798154
H                 7.1734702765       -0.9905622664       -4.4898145275
H                 6.4124015868       -2.4018088000       -3.7143712703
H                 7.2942861957       -1.2032494792       -2.7256667706
H                 5.6720799238       -2.3771794699       -1.4695466008
H                 4.0028638245       -2.5937496042        0.3042195555
H                 0.9549793432       -0.5394803417        0.3713435007
H                 0.7664750724       -1.3850328799        3.5671294597
H                -1.1695681606       -2.8622560956        3.1838750923
H                -0.6248874840       -3.1867728944        1.5271296368
H                 0.3462826441       -3.7394908291        2.9049274027
H                -0.8792779722       -0.1405859189       -1.4067805439
H                -0.8052856465       -2.4504938537       -2.2564351588
H                -2.1984401387       -4.1977078067       -1.2069681767
H                -3.6516813110       -3.6255603179        0.7032055564
H                -3.7077370314       -1.3171607732        1.5640319607
H                -3.4128326471        0.6694592309       -1.8426881450
H                -4.3574842916        2.5430184964       -3.1187337028
H                -4.4277602303        4.8079868457       -2.1421304245
H                -3.5480422374        5.1814707160        0.1533060140
H                -1.9611342817        1.9535769035        4.5242463059
H                -0.3729181888        2.3495098886        3.8329017759
H                -1.6015041005        3.6257459014        4.0179728437
H                 0.3160424447        2.2231642126       -1.9670787482
H                 1.1157585083        1.3233416182       -3.2822063291
H                 2.0589735686        2.4419355304       -2.2582793211
H                 3.4088777418        0.7769219455       -3.2319019951


