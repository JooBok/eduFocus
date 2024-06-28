import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.backend import clear_session
from model import CNN_reg, CNN_classif
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import argparse
import json
from utils import configure_gpu, load_ADHD_classification_sub_info, load_all_sub_info, load_X_input_files, padding_zeros, compute_mean_std

# 모델을 사전 학습하는 함수
def model_pre_train(test_user, video_indx, sal_model, remove_input_channel, pad_len, sub_info_path, input_dir, gpu):
    print(f'start evaluate with video{video_indx}, saliency mode: {sal_model}')
    dict_video_indx_name_mapping = {1: 'Diary_of_a_Wimpy_Kid_Trailer', 2: 'Fractals', 3: 'Despicable_Me', 4:'The_Present'}
    video_name = dict_video_indx_name_mapping.get(video_indx)
    # 피험자 정보 불러오기
    df_sub, sub_list = load_all_sub_info(video_name, sub_info_path)
    # Swan 점수가 없는 피험자 제외
    df_sub = df_sub[df_sub['SWAN_Total'].notna()]
    # 테스트 데이터에 있는 피험자 제외
    df_sub = df_sub[~df_sub.Patient_ID.isin(test_user)]
    user_list = df_sub.Patient_ID.values.tolist()

    # 피험자의 입력 파일과 해당 라벨 불러오기
    # 이상한 수의 응시를 가진 사용자는 제외됨
    # 응시 수의 최대값 반환
    X, Y, user_list, max_len = load_X_input_files(input_dir,
                                                    video_indx,
                                                    user_list,
                                                    df_sub,
                                                    label='reg',
                                                    remove_input_channel=remove_input_channel)

    # Swan 점수의 범위가 유사하게 포함되도록 훈련 및 테스트 데이터 생성
    sign = Y.copy()
    sign[sign>=0.5] = 1
    sign[sign<0.5] = 0

    n_folds=10
    # 모델 훈련을 위한 훈련 및 검증 데이터 분할
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for train_idx, val_idx in tqdm(skf.split(X, sign)):
        break

    X_train_net = [X[idx] for idx in train_idx]
    X_val_net = [X[idx] for idx in val_idx]
    Y_train_net, Y_val_net = Y[train_idx,:], Y[val_idx,:]

    # z-score 정규화를 위한 평균 및 표준편차 계산
    mean, std = compute_mean_std(X_train_net, remove_input_channel)

    # 정규화 적용
    X_train_net = [(x - mean)/std for x in X_train_net]
    X_val_net = [(x - mean)/std for x in X_val_net]

    # 제로 패딩
    if pad_len>max_len:
        max_len = pad_len

    X_train_net = padding_zeros(X_train_net,  pad_len=max_len)
    X_val_net = padding_zeros(X_val_net,  pad_len=max_len)

    # tensorflow 세션 초기화
    clear_session()
    print('Pre-train the model.')
    DNN_model = CNN_reg(seq_len=X_train_net.shape[1],channels=X_train_net.shape[2])
    model = DNN_model.train(X_train_net, Y_train_net, X_val_net, Y_val_net)

    return model, max_len, mean, std

# 모델 평가 함수
def evaluate(video_indx, sal_model, remove_input_channel, pretrain, sub_info_path, input_dir, n_rounds, n_folds, results_save_dir, save_weights_flag, gpu):
    print(f'start evaluate with video{video_indx}, saliency mode: {sal_model}')

    dict_video_indx_name_mapping = {1: 'Diary_of_a_Wimpy_Kid_Trailer', 2: 'Fractals', 3: 'Despicable_Me', 4:'The_Present'}
    video_name = dict_video_indx_name_mapping.get(video_indx)

    # ADHD 분류 설정을 위한 피험자 정보 불러오기
    df_sub_classif, user_list = load_ADHD_classification_sub_info(video_name, sub_info_path)
    # 피험자의 입력 파일과 해당 라벨 불러오기
    # 이상한 수의 응시를 가진 사용자는 제외됨
    # 응시 수의 최대값 반환
    X, Y, user_list, max_len = load_X_input_files(input_dir,
                                                    video_indx,
                                                    user_list,
                                                    df_sub_classif,
                                                    label='binary',
                                                    remove_input_channel=remove_input_channel)
    print('number of ADHD vesus control people:', np.count_nonzero(Y), Y.shape[0]-np.count_nonzero(Y))

    auc_rounds = []
    fold_counter = 1
    for i in range(n_rounds):
        aucs=[]
        # 교차 검증을 위한 n 폴드로 사용자 분할
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
        fold_idx=1
        for train_idx, test_idx in tqdm(skf.split(X, Y)):
            print('Starting evaluation fold {}/{}...'.format(fold_idx, n_folds))
            X_train = [X[idx] for idx in train_idx]
            X_test = [X[idx] for idx in test_idx]
            Y_train, Y_test = Y[train_idx,:], Y[test_idx,:]
            test_user = [user_list[i] for i in test_idx]
            train_user = [user_list[i] for i in train_idx]

            # 모델 훈련을 위한 훈련 및 검증 데이터 분할:
            skf_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
            for train_index, val_index in skf_val.split(X_train, Y_train):
                # 단일 폴드만 평가
                break

            X_train_net = [X_train[idx] for idx in train_index]
            X_val_net = [X_train[idx] for idx in val_index]
            Y_train_net, Y_val_net = Y_train[train_index,:], Y_train[val_index,:]
            val_user = [train_user[i] for i in val_index]

            # tensorflow 세션 초기화
            clear_session()
            configure_gpu(gpu)
            print('Build and train model.')
            if pretrain==True:
                # 모델 사전 학습
                # 훈련 중 데이터가 보이지 않도록 테스트 및 검증 데이터를 제외
                exclude_user = test_user + val_user
                pretrain_model, max_len, mean, std = model_pre_train(test_user = exclude_user,
                                                                                video_indx=video_indx,
                                                                                sal_model = sal_model,
                                                                                remove_input_channel = remove_input_channel,
                                                                                pad_len=max_len,
                                                                                sub_info_path=sub_info_path,
                                                                                input_dir=input_dir,
                                                                                gpu=gpu)


            else:
                pretrain_model=None
                # z-score 정규화를 위한 평균 및 표준편차 계산
                mean, std = compute_mean_std(X_train_net, remove_input_channel)

            # 정규화 적용
            X_train_net = [(x - mean)/std for x in X_train_net]
            X_val_net = [(x - mean)/std for x in X_val_net]
            X_test = [(x - mean)/std for x in X_test]

            X_train_net = padding_zeros(X_train_net,  pad_len=max_len)
            X_val_net = padding_zeros(X_val_net,  pad_len=max_len)
            X_test = padding_zeros(X_test,  pad_len=max_len)

            DNN_model = CNN_classif(seq_len=X_train_net.shape[1],channels=X_train_net.shape[2])
            hist = DNN_model.train(X_train_net, Y_train_net,
                                    X_val_net, Y_val_net,
                                    pretrain_model = pretrain_model,
                                    save_weights_flag = save_weights_flag,
                                    fold_counter = fold_counter)

            pred_test = np.squeeze(DNN_model.model.predict([X_test]))
            fpr, tpr, thresholds = metrics.roc_curve(Y_test, pred_test)
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc.tolist())
            print(aucs)
            fold_idx +=1
            fold_counter +=1
        auc_rounds.append(aucs)

    print('Mean AUC:', np.mean(auc_rounds))
    print('Standanrd error:', np.std(auc_rounds)/(n_rounds*n_folds))
    # 결과 저장
    dic={'auc': auc_rounds}
    cur_save_path = results_save_dir + f'res_Video{video_indx}_pretrain{pretrain}_remove{remove_input_channel}_{num_folds}folds_{num_iter}iter.json'
    isExist = os.path.exists(cur_save_path)
    if not isExist:
        os.makedirs(cur_save_path)
    with open(cur_save_path, 'w') as fp:
        json.dump(dic, fp, indent = 4)

    print('Finish evaluation, results saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hyperparameter tuning')
    parser.add_argument(
        '--video_indx',
        help='video_index',
        type=int,
        default=2  # 분석할 비디오 인덱스로 변경
    )

    parser.add_argument(
        '--sal_model',
        help='saliency model name',
        type=str,
        default='DeepGazeII'  # 사용할 모델로 변경
    )

    parser.add_argument(
        '--remove_input_channel',
        help='When using all input channels set to NA. For the ablation study, one of the four input channels is removed, the valid input channels to be removed are loc, dur, sal.',
        type=str,
        default='NA'  # 필요시 제거할 채널로 변경
    )

    parser.add_argument(
        '--pre_train',
        help='if do pretraining',
        type=int,
        default=True
    )

    parser.add_argument(
        '--sub_info_path',
        help='Directory for saving subject information',
        type=str,
        default='./Data/sub_info/'  # 실제 피험자 정보 경로로 변경
    )
     
    parser.add_argument(
        '--input_dir',
        help='Directory for saved input files',
        type=str,
        default='./Data/X_input/'  # 실제 input 데이터 경로로 변경
    )

    parser.add_argument(
        '--num_iter',
        help='Number of iterations to run',
        type=int,
        default=10
    )

    parser.add_argument(
        '--num_folds',
        help='Number of splitting folds for cross-validation',
        type=int,
        default=10
    )

    parser.add_argument(
        '--results_save_dir',
        help='directory for saving results',
        type=str,
        default='./results/'  # 실제 결과 저장 경로로 변경
    )

    parser.add_argument(
        '--save_weights_flag',
        help='Flag to save weights for all the models',
        type=int,
        default=0
    )

    parser.add_argument(
        '--gpu',
        help='gpu_index',
        type=int,
        default=6
    )

    args = parser.parse_args()
    video_indx = args.video_indx
    sal_model = args.sal_model
    remove_input_channel = args.remove_input_channel
    assert remove_input_channel in ["NA", "loc", "dur", "sal"], "No such input channels to be removed"
    pre_train = args.pre_train
    sub_info_path = args.sub_info_path
    input_dir = args.input_dir
    num_iter = args.num_iter
    num_folds = args.num_folds
    results_save_dir = args.results_save_dir
    save_weights_flag = args.save_weights_flag
    gpu_indx = args.gpu

    evaluate(video_indx=video_indx,
                sal_model=sal_model,
                remove_input_channel=remove_input_channel,
                pretrain=pre_train,
                sub_info_path=sub_info_path,
                input_dir=input_dir,
                n_rounds=num_iter,
                n_folds=num_folds,
                results_save_dir=results_save_dir,
                save_weights_flag=save_weights_flag,
                gpu=gpu_indx)
