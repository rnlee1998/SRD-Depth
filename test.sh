for((i=12;i<20;i++))
do 
	CUDA_VISIBLE_DEVICES=0 nohup python -u evaluate_depth.py --load_weights_folder /mnt/data/liran/workdir/monovit/save_models/$*/mono_epc1_fa_ck2/models/weights_$i/ --eval_mono >>test_result_$*_epc1_fa_ck2.log 2>&1 &	
	sleep 25s   
	echo $i "test finish"
done

for((i=12;i<20;i++))
do 
	CUDA_VISIBLE_DEVICES=0 nohup python -u evaluate_depth.py --load_weights_folder /mnt/data/liran/workdir/monovit/save_models/$*/mono_epc1_fa_ck3/models/weights_$i/ --eval_mono >>test_result_$*_epc1_fa_ck3.log 2>&1 &	
	sleep 25s   
	echo $i "test finish"
done




