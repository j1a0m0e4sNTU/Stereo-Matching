python3 main.py train -id "target" -info "Reproduce PSMNET"
# python3 main.py train -epoch_num 20 -id "Net0_v1" -info "baseline -- extract feature then stack" 
# python3 main.py train -epoch_num 20 -id "Net1_v1" -info "baseline -- stack then extract feature" 
# python3 main.py train -epoch_num 20 -id "Net_bf_v1" -info "Brute Force -- fit left image to its disparity" 
# python3 main.py train -epoch_num 20 -id "SNet_0_v1" -info "multiresolution feature + resudual blocks" 
# python3 main.py train -epoch_num 20 -id "SNet_0_v2" -info "add SoftMax when compute disp_prob" 