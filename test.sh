for((i=12;i<20;i++))
do 
	CUDA_VISIBLE_DEVICES=1 nohup python -u evaluate_depth.py --load_weights_folder /mnt/data/liran/workdir/monovit/save_models/$*/mono_epc1_woCutflip/models/weights_$i/ --eval_mono >>test_result_$*_epc1_woCutflip.log 2>&1 &	
	sleep 25s   
	echo $i "test finish"
done

for((i=12;i<20;i++))
do 
	CUDA_VISIBLE_DEVICES=1 nohup python -u evaluate_depth.py --load_weights_folder /mnt/data/liran/workdir/monovit/save_models/$*/mono_epc1_woCutflip2/models/weights_$i/ --eval_mono >>test_result_$*_epc1_woCutflip2.log 2>&1 &	
	sleep 25s   
	echo $i "test finish"
done

for((i=12;i<20;i++))
do 
	CUDA_VISIBLE_DEVICES=1 nohup python -u evaluate_depth.py --load_weights_folder /mnt/data/liran/workdir/monovit/save_models/$*/mono_epc1_woCutflip3/models/weights_$i/ --eval_mono >>test_result_$*_epc1_woCutflip3.log 2>&1 &	
	sleep 25s   
	echo $i "test finish"
done

for((i=12;i<20;i++))
do 
	CUDA_VISIBLE_DEVICES=1 nohup python -u evaluate_depth.py --load_weights_folder /mnt/data/liran/workdir/monovit/save_models/$*/mono_epc1_woCutflip4/models/weights_$i/ --eval_mono >>test_result_$*_epc1_woCutflip4.log 2>&1 &	
	sleep 25s   
	echo $i "test finish"
done


