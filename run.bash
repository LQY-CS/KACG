
#Cora
# python tran_train.py --num_layers=2 --hidden_dim=64 --dropout_t=0.5 --dropout_s=0.4 --learning_rate=0.05 --weight_decay=0.0001 --max_epoch=200 --seed=6959682 --data_mode=0 --tau_0=1.2 --tau_1=1.5 --topk=4 --alpha=0.9 --l_alpha=1.7403289773124893 --l_beta=3.101971911673752 --l_gamma=0.5 --beta=0.9593733958602548 --warm_up=40

#CiteSeer
python tran_train.py --num_layers=2 --hidden_dim=64 --dropout_t=0.4 --dropout_s=0.6 --learning_rate=0.05 --weight_decay=0.005 --max_epoch=200 --seed=0 --data_mode=1 --tau_0=1 --tau_1=1.5 --topk=1 --alpha=0.87 --l_alpha=4.91 --l_beta=9.71 --l_gamma=0.5 --beta=0.82 --warm_up=20

#PubMed
# python tran_train.py --num_layers=2 --hidden_dim=64 --dropout_t=0.6 --dropout_s=0.3 --learning_rate=0.05 --weight_decay=0.0001 --max_epoch=200 --seed=620207 --data_mode=2 --tau_0=0.2 --tau_1=0.5 --topk=3 --alpha=0.5 --l_alpha=5.399971923486876 --l_beta=5.39857795662101 --l_gamma=0.5 --beta=0.7280016792663153 --warm_up=50

#C-CS
# python tran_train.py --num_layers=2 --hidden_dim=64 --dropout_t=0.5 --dropout_s=0.5 --learning_rate=0.05 --weight_decay=0.005 --max_epoch=200 --seed=0 --data_mode=3 --tau_0=1 --tau_1=1.5 --topk=2 --alpha=0.97 --l_alpha=9.81 --l_beta=7.61 --l_gamma=0.5 --beta=0.48 --warm_up=30

#C-Phy
# python tran_train.py --num_layers=2 --hidden_dim=64 --dropout_t=0.4 --dropout_s=0.7 --learning_rate=0.05 --weight_decay=0.0005 --max_epoch=200 --seed=400301 --data_mode=4 --tau_0=0.5 --tau_1=2.0 --topk=3 --alpha=0.7 --l_alpha=2.299174831607293 --l_beta=9.820280768198144 --l_gamma=0.5 --beta=0.7241664927773883 --warm_up=30

#A-Photo
# python tran_train.py --num_layers=2 --hidden_dim=64 --dropout_t=0.4 --dropout_s=0.7 --learning_rate=0.005 --weight_decay=0.001 --max_epoch=200 --seed=0 --data_mode=5 --tau_0=1 --tau_1=2.5 --topk=1 --alpha=0.88 --l_alpha=0.31 --l_beta=95.71 --l_gamma=0.5 --beta=0.01 --warm_up=10


#A-Computers
# python tran_train_finnal.py --num_layers=2 --hidden_dim=64 --dropout_t=0.4 --dropout_s=0.2 --learning_rate=0.005 --weight_decay=0.005 --max_epoch=200 --seed=353844 --data_mode=6 --tau_0=0.7 --tau_1=2.0 --topk=1 --alpha=0.6 --l_alpha=2.859645946963792 --l_beta=0.0356461479244465 --l_gamma=0.5 --beta=0.7097546519516305 --warm_up=20