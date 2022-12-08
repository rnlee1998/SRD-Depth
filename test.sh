for((i=12;i<20;i++))
do 
	CUDA_VISIBLE_DEVICES=0 nohup python -u evaluate_depth.py --load_weights_folder /mnt/data/liran/workdir/monovit/save_models/$*/mono_epc1_fa_swinT/models/weights_$i/ --eval_mono >>test_result_$*_epc1_fa_swinT.log 2>&1 &	
	sleep 25s   
	echo $i "test finish"
done




