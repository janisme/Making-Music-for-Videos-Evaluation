from shotdetectk import shotd  # Adjust import statement based on actual file and function location
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse
import shutil
import os
import os.path as osp
import csv
import random
import torch
import collections


def v2frame():
    #convert vedio to frame
    preprocess_dir = "../data/demo/"
    
    if os.path.exists(preprocess_dir) and os.path.isdir(preprocess_dir):
        shutil.rmtree(preprocess_dir)

    os.makedirs(preprocess_dir, exist_ok = True, mode=0o777)
    os.makedirs(osp.join(preprocess_dir,"video"), exist_ok = True, mode=0o777)

    vidop_ori_path = "../vmcpdata/simple_mp4"
    video_save_path = "../data/demo/video"

    shutil.move(osp.join(vidop_ori_path,os.listdir(vidop_ori_path)[0]),osp.join(video_save_path,"demo.mp4"))


# Create a function to mimic argparse's namespace object
def create_args():
    parser = argparse.ArgumentParser("Single Video ShotDetect")
    parser.add_argument('--video_path', type=str,
                        default="../data/demo/video/demo.mp4",
                        help="path to the video to be processed")
    parser.add_argument('--save_data_root_path', type=str,
                        default="../data/demo",
                        help="path to the saved data")
    parser.add_argument('--print_result',    action="store_true")
    parser.add_argument('--save_keyf',       action="store_true", default=True)
    parser.add_argument('--save_keyf_txt',   action="store_true")
    parser.add_argument('--split_video',     action="store_true")
    parser.add_argument('--keep_resolution', action="store_true")
    parser.add_argument('--avg_sample',      action="store_true")
    parser.add_argument('--begin_time',  type=float, default=None,  help="float: timecode")
    parser.add_argument('--end_time',    type=float, default=120.0, help="float: timecode")
    parser.add_argument('--begin_frame', type=int,   default=None,  help="int: frame")
    parser.add_argument('--end_frame',   type=int,   default=1000,  help="int: frame")
    args, _ = parser.parse_known_args()
    return args

#get dic of list {idx: [v_num, text_simple, text_30, text_50]}
def read_wording(csv_path):
    wording_list= collections.defaultdict()
    with open(csv_path, mode = 'r') as f:
        csv_reader = csv.reader(f)
        
        #skip the header
        next(csv_reader, None)
        vi2i = collections.defaultdict()
        idx =0
        for row in csv_reader: 
            wording_list[idx] = row
            vi2i[row[0]] =idx
            idx +=1

    return wording_list, vi2i

# return accuracy and average ranking per film
def v_t_simi(wording_list, model,k, v_idx, g_idx, list_len, target):

    dir_path = '../data/demo/shot_keyf/demo'
    accuracy = 0.0
    total_rk = 0.0
    ct =0.0

    for filename in os.listdir(dir_path):
        ct+=1
        #run thru all the file
        file_path = os.path.join(dir_path, filename)

        if file_path.lower().endswith('.jpg'):

            # use for percision @ K
            # random_list also use for reference
            random_list = random.sample(range(0, list_len), k)

            #make sure correct is in it
            if g_idx not in random_list:
                random_list[-1] = g_idx  
            # print(random_list) 
            #text_simple is a list of list [[v_num, text_simple, text_30, text_50]]
            text_simple =[]
            ct_random_list=0
            for idx in random_list:
                # print(wording_list)
                txt = wording_list[idx][target]
                text_simple.append(txt)
                if idx == g_idx:
                    #在random list中第幾個是target
                    target_rl_idx = ct_random_list
                ct_random_list +=1

            with Image.open(file_path) as img:
                
                inputs = processor(text=text_simple, images=img, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                # print(logits_per_image)
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                # print("all prob: ",probs)

                max_value, idx = torch.max(probs,-1)
                
                p_g_id = random_list[idx.item()]
                p_v_id = wording_list[p_g_id][0]
                print(f'Prob: {max_value.item()}, g_id = {p_g_id}, v_id = {p_v_id}')
                print(f'wording = {text_simple[idx.item()]}')
                if p_g_id == g_idx:
                    accuracy += 1
                #找出答案排第幾    
                rank =1
                # print(probs[0])
                # print(probs[0][target_rl_idx].item())
                for prob in probs[0]:
                    if prob.item() > probs[0][target_rl_idx].item():
                        rank +=1
                
                total_rk += rank 
                # print(total_rk)
                
    print(f'total accuracy for this clip: {accuracy/ct}')
    return accuracy/ct, total_rk/ct






# Call the main function with created args >>>video2desc.txt
if __name__ == '__main__':
    target_dir = "../vmcpdata/simple_mp4"
    source_dir = "../vmcpdata/simple_mp4 copy"
    csv_path = '../vmcpdata/wording.csv'
    #generation model(1:simple, 2:30, 3:50)
    target = 3
    #percision k
    k = 13
    
    vidop_ori_path = target_dir
    
    # read wording list
    wording_list, vi2i = read_wording(csv_path)
    print("wording_list complete! =======")
    print("Number of wording: ", len(wording_list))

    # load model
    print("Loading model ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Model loading complete! =======")

    for sample in range(target):
        # make sample dir
        try:
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
            print("Directory copied and overwritten successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")

        #check stat
        v_len  = len(os.listdir(vidop_ori_path))
        list_len = len(wording_list)
        print(f'Total video: {v_len}, total wording: {list_len}, p @ {k}, sample_target = {sample+1}')


        total_ac =0.0
        total_rk = 0.0
        ct = 0  #number of word
        for i in range(v_len):
                
            # get the video id
            v_idx = os.listdir(vidop_ori_path)[0].split("Q")[1].split(".")[0]
            g_idx = vi2i[v_idx]
            ct+=1
            print(f'====== # {ct} / {v_len} ')
            print("Process vedio: vedio id = ", v_idx, " generated id = ", g_idx)

                
            # convert vedio to frame
            v2frame()
            args = create_args()
            data_root = args.save_data_root_path
            shotd(args, data_root)
            fram_dir = "../data/demo/shot_keyf/demo"
            num_clip = len(os.listdir(fram_dir))
            num_shot = int(os.listdir(fram_dir)[-1].split("_")[1])
            print(f'%%%%num of clip: {num_clip}, num of shot: {num_shot+1}')

            #cal each frames accuracy and avrage
            ac_per_v, ranking =  v_t_simi(wording_list, model,k, v_idx, g_idx, list_len, sample+1)
            print("&&&&&&&&", ranking)
            total_ac += ac_per_v
            total_rk +=  ranking

        print(f'********* Testing sample {sample+1} *******')
        print("======Total accuracy: ", total_ac/ct)
        print("======Total ranking: ", total_rk/ct)



            
