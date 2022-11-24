%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/135511190_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/135511190_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -2.5864355162       -3.0569645753       -1.9289143889
C                -1.3043420797       -2.5517161135       -2.6166012132
C                -0.2571565961       -2.0826313358       -1.6337244536
C                -0.0188196913       -0.7855062030       -1.2482158104
C                -0.7697761717        0.4146061221       -1.8086470197
C                -1.3157681133        1.3356734022       -0.7397076954
C                -2.5820640760        1.1538037700       -0.1800828629
F                -3.3314656247        0.1186963096       -0.6268826550
C                -3.1204562339        1.9817434495        0.7974426416
C                -2.3574221171        3.0587451662        1.2487846202
C                -1.0861736590        3.2929498689        0.7237995925
C                -0.5987158308        2.4299224731       -0.2509752342
F                 0.6289465856        2.6718216822       -0.7622578880
N                 0.9579225558       -0.4225227805       -0.3406148838
C                 1.6897058895       -1.3568760601        0.1815094763
S                 3.0123889405       -1.0120794967        1.3178611926
C                 2.7665188470        0.8040433862        1.6387403078
C                 1.6699332547        1.0361410418        2.6780906629
C                 4.1213542897        1.3776994991        2.0622002713
N                 1.5187878776       -2.6749472172       -0.1220705757
C                 0.5474202242       -3.1433019833       -1.0326653289
O                 0.4520160681       -4.3462839830       -1.2474245193
H                -2.3537662863       -3.8900289079       -1.2586339970
H                -3.0579748008       -2.2604301287       -1.3453450477
H                -3.3075216730       -3.4123357912       -2.6740277510
H                -0.8701499626       -3.3737402709       -3.1960454817
H                -1.5517810128       -1.7578682534       -3.3280664500
H                -0.0665667874        0.9840957179       -2.4271444888
H                -1.5882531293        0.0889897777       -2.4503632822
H                -4.1134607865        1.7766746251        1.1817730496
H                -2.7567038647        3.7207276252        2.0111037318
H                -0.4743242842        4.1271327549        1.0493214470
H                 2.4552039988        1.2210785143        0.6780238156
H                 0.7167590595        0.6199795727        2.3426042370
H                 1.5316672151        2.1143162264        2.8297943788
H                 1.9329061662        0.5859570928        3.6416783390
H                 4.4864406993        0.9135742756        2.9861204459
H                 4.8795446563        1.2417822873        1.2851471349
H                 4.0136924946        2.4518861469        2.2524512167
H                 2.0878904751       -3.3948056873        0.3119674658


