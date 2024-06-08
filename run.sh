python main.py +hydra/launcher=grogu math_operator=+,s5  +exp=ff,f train_data_pct=20,40,60,80 -m 

1-12 - matrix not working

python main.py +hydra/launcher=matrix math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+-  +exp=ff,f train_data_pct=50 -m 


python main.py +hydra/launcher=grogu math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+-,reverse  exp=ff,f train_data_pct=50,40 weight_decay=1.0 group=m10 -m
python main.py +hydra/launcher=grogu math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+-,reverse  exp=ff train_data_pct=50  group=m7 weight_decay=1.0 d_model=256 n_layers=4 -m

python main.py +hydra/launcher=grogu math_operator=+


python main.py math_operator=+   do_tta=True  cc_coef=0.0 tta_coef=0.0 inv_coef=0.0 +hydra/launcher=matrix -m




python main.py +hydra/launcher=grogu math_operator=-,s5,+,**3+,* train_data_pct=60 exp=ff inv_coef=0.1,0.5,1.0 -m
python main.py +hydra/launcher=grogu math_operator=-,s5,+,**3+,* train_data_pct=60 exp=ff do_tta=True tta_coef=0.1,0.5,1.0 -m
python main.py +hydra/launcher=grogu math_operator=-,s5,+,**3+,* train_data_pct=60 exp=ff cyclic_consistency=True cc_coef=0.1,0.5,1.0 -m




python main.py +hydra/launcher=grogu math_operator=-,s5,+,**3+,* train_data_pct=60 exp=ff do_tta=True tta_coef=0.1 inv_coef=0.1 group=m11 weight_decay=1.0 -m
python main.py +hydra/launcher=grogu math_operator=-,s5,+,**3+,* train_data_pct=60 exp=ff cyclic_consistency=True cc_coef=0.1 inv_coef=0.1 group=m11 weight_decay=1.0 -m
python main.py +hydra/launcher=grogu math_operator=-,s5,+,**3+,* train_data_pct=60 exp=f weight_decay=1.0 group=m11 -m
python main.py +hydra/launcher=grogu math_operator=-,s5,+,**3+,* train_data_pct=60 exp=ff inv_coef=0.1 weight_decay=1.0 group=m11 -m
python main.py +hydra/launcher=grogu math_operator=-,s5,+,**3+,* train_data_pct=60 exp=ff cyclic_consistency=True cc_coef=0.1 inv_coef=0.1 group=m11 -m
python main.py +hydra/launcher=grogu math_operator=-,s5,+,**3+,* train_data_pct=60 exp=ff do_tta=True tta_coef=0.05 inv_coef=0.1 group=m11 -m


python main.py +hydra/launcher=grogu math_operator=-,s5,+,**3+,* train_data_pct=60 exp=ff do_tta=True tta_coef=0.1 inv_coef=0.0 group=m12 -m


python main.py +hydra/launcher=matrix math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+-,reverse train_data_pct=30,40,50 exp=ff do_tta=True tta_coef=0.03 inv_coef=0.1 weight_decay=1.0 group=m12 -m



python main.py +hydra/launcher=grogu math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+-,reverse train_data_pct=30,40,50 exp=ff do_tta=True tta_coef=0.03 inv_coef=0.1 weight_decay=1.0 group=m12 -m


python main.py +hydra/launcher=matrix math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+-,reverse train_data_pct=30,40,50 exp=f weight_decay=1.0 group=m12 -m


python main.py +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+,-,reverse train_data_pct=30,40,50 exp=f weight_decay=1.0 group=m12 -m

python main.py +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+,- train_data_pct=30 exp=f weight_decay=1.5,2.0,2.5 group=m12 -m


python main.py +hydra/launcher=matrix math_operator=- train_data_pct=30 exp=f  group=tmp -m

+,-,**2+,+-,**3+,/


python main.py +hydra/launcher=matrix math_operator=s5,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+* train_data_pct=30 exp=f weight_decay=1.0 group=m12 -m


python main.py +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+,- train_data_pct=20 exp=ff do_tta=True tta_coef=0.03 inv_coef=0.1 weight_decay=1.0 group=m12 -m 


python main.py +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+,- exp=ff train_data_pct=30 do_tta=True tta_coef=0.03 inv_coef=0.1 weight_decay=1.0 group=m13 -m 

python main.py +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+,- exp=f train_data_pct=30 exp=f weight_decay=2.0 group=m13 -m 




python main.py +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+,- train_data_pct=20 exp=f weight_decay=2.0  group=m12 d_model=384 -m


python main.py +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+,-  inverse_train=True forward_train=True  val_batchify=False train_data_pct=40 steps_to_tta=2000,5000,1000 -m


python main.py  +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+ train_data_pct=30 exp=f  weight_decay=1.0 multi_task=True multi_coef=1.0,0.1 group=m13 -m

python main.py  +hydra/launcher=grogu math_operator=+,/,**3+,+-,**2+ train_data_pct=30 exp=f  weight_decay=1.0 multi_task=True multi_coef=2.0 group=m13 -m

python main.py  +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+ train_data_pct=30 exp=f  weight_decay=1.0 multi_task=True multi_coef=2.0 group=m13 -m


python main.py  +hydra/launcher=matrix math_operator=- math_operator_2=+ train_data_pct=30 exp=f  weight_decay=1.0 multi_task=True multi_coef=0.03,0.3 group=m13 -m


python main.py  +hydra/launcher=grogu math_operator=- math_operator_2=+ train_data_pct=30 exp=f  weight_decay=1.0 multi_task=True  multi_coef=1.0,0.1 group=m13 -m


+,/,**3+,+-,**2+


python main.py  +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+ train_data_pct=30 exp=f  weight_decay=1.0 multi_task=True  group=m13 -m


python main.py +hydra/launcher=matrix math_operator=+,/,**3+,+-,**2+,- exp=ff train_data_pct=30 do_tta=True tta_coef=0.03 inv_coef=0.1 weight_decay=1.0 max_steps=1000 group=m14 -m 



python main.py  +hydra/launcher=grogu  math_operator=+,/,**3+,+-,**2+,- train_data_pct=10,20,30,40,50,60,70,80,90 exp=ff do_tta=True tta_coef=0.03 inv_coef=0.1 weight_decay=1.0 group=m15 -m 


python main.py  +hydra/launcher=matrix  math_operator=+,/,**3+,+-,**2+,- train_data_pct=10,20,30,40,50,60,70,80,90 exp=f  weight_decay=1.0 group=m15 -m 

python main.py math_operator=+ train_data_pct=10 exp=ff do_tta=True tta_coef=0.03 inv_coef=0.1 weight_decay=1.0 group=m15 -m 

python main.py  +hydra/launcher=matrix  math_operator=**3+,**2+ train_data_pct=10,20,30,40,50,60,70,80,90 exp=f  weight_decay=1.0 group=m16 -m 


python main.py  +hydra/launcher=matrix  math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+- train_data_pct=50 exp=ff group=m17 -m
python main.py  +hydra/launcher=grogu  math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+- train_data_pct=50 exp=f group=m17 -m


python main.py math_operator=+ train_data_pct=50 exp=ff  group=vis

python main.py math_operator=+,/,**3+,+-,**2+,- train_data_pct=50 exp=ff  group=vis f_coef=0.0 tta_coef=0.0 plot_pca_last_layer=True +hydra/launcher=matrix -m 

python main.py math_operator=+,/,**3+,+-,**2+,- train_data_pct=50 exp=f  group=vis  plot_pca_last_layer=True +hydra/launcher=grogu -m 
python main.py math_operator=+,/,**3+,+-,**2+,- train_data_pct=50 exp=ff  group=vis  plot_pca_last_layer=True +hydra/launcher=grogu -m 

squeue -O Partition,UserName,JobID,mem-per-tres,cpus-per-task,tres-per-job,tres-per-node,TimeLeft,ReqNodes,NodeList,Name | grep inver | wc -l



sort,reverse,copy,pfactor,2x,x**3,2x+1,interleaved_halves,reverse_pool,k_shift,random_swaps,idx_add,caesarcipher_permutev1,caesarcipher,permutev1,permutev2,permutev3,strdeletev1,strdeletev2,pfactor,2x,x**3,2x+1,x+11


python main.py  +hydra/launcher=grogu  math_operator=sort,reverse,copy,pfactor,2x,x**3,2x+1,interleaved_halves,reverse_pool,k_shift,random_swaps,idx_add,caesarcipher_permutev1,caesarcipher,permutev1,permutev2,permutev3,strdeletev1,strdeletev2,pfactor,2x,x**3,2x+1,x+11 train_data_pct=50 exp=ff group=m19 -m


2x
2x+1
pfactor
pfactor
x**3

python main.py  +hydra/launcher=grogu  math_operator=pfactor,x**3 train_data_pct=50 exp=ff max_context_len=100 group=m19 -m


python main.py  +hydra/launcher=grogu  math_operator=pfactor,x**3 train_data_pct=50 exp=ff max_context_len=100 group=m19 -m
+hydra/launcher=matrix  -m




python main.py  +hydra/launcher=matrix  math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+- train_data_pct=20 exp=ff group=m26 -m
python main.py  +hydra/launcher=matrix  math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+- train_data_pct=40 exp=f group=m26 -m


python main.py  +hydra/launcher=grogu  math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+- train_data_pct=20 exp=f group=m26 -m
python main.py  +hydra/launcher=grogu  math_operator=+,s5,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+- train_data_pct=40 exp=ff group=m26 -m



python main.py exp=f math_operator=+,/,**3+,+-,**2+,-,* math_operator_2=x**3+x*y**2+y_mod_97 multi_task=True train_data_pct=50 visualize=True save_activations=True group=multi3 +hydra/launcher=grogu -m 

python main.py exp=ff math_operator=+,/,**3+,+-,**2+,-,* train_data_pct=50 visualize=True save_activations=True group=multi3 +hydra/launcher=matrix -m 


python main.py exp=ff math_operator=+,/,**3+,+-,**2+,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+- group=inv_3 +hydra/launcher=grogu -m

python main.py exp=ff math_operator=+,/,**3+,+-,**2+,-,*,/,**2+,**3+,x**2+y**2_mod_97,x**2+y**2+x*y_mod_97,x**2+y**2+x*y+x_mod_97,x**3+x*y_mod_97,x**3+x*y**2+y_mod_97,s5conj,s5aba,+*,+- group=inv_4 train_data_pct=40 +hydra/launcher=grogu -m

