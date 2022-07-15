#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.cuda import FloatTensor, LongTensor
import numpy as np

class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).

    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"
        dpath = "/".join(file_path.split("/")[0:-1])
        self.dq2c = pd.read_pickle(os.path.join(dpath, "dq2c.pkl"))
        # print(self.dq2c)

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks", "orics", "orisms", "futuresms"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            if key == "oriqs":
                cursm = self.dori["orisms"][index]
                dcur[key] = self.dori[key][index][:-1] * cursm
                dcur["shft_"+key] = self.dori[key][index][1:] * cursm
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        # if index == 30:
        #     print(f"qs: {dcur['qseqs']}")
        #     print(f"oriqs: {dcur['oriqs']}")
        #     assert False
        
        dcur["orics"] = self.dori["orics"][index][:-1]
        dcur["shft_orics"] = self.dori["orics"][index][1:]
        dcur["orisms"] = self.dori["orisms"][index]

        dcur["futuresms"] = self.dori["futuresms"][index]

        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    def __generate_future__(self, cs, rs, numc):
        deachskill = dict()
        cs, rs = torch.LongTensor(cs), torch.LongTensor(rs)
        for i in range(0, numc):
            curc, curr = cs[cs==i], rs[cs==i]
            for j in range(0, len(curc)):
                right = len(curr[j+1:][curr[j+1:]==1])
                total = len(curc) - j - 1
                if total > 0:
                    correct_rate = right / total
                else:
                    correct_rate = 0 # skill的最后一个预测默认是0，sm会筛掉
                # print(f"i: {i}, j: {j}, curc: {curc.shape}, future right: {right}, total: {total}, rate: {correct_rate}")
                deachskill.setdefault(i, list())
                deachskill[i].append([j, correct_rate])
        cs, rs = cs.tolist(), rs.tolist()
        allrates, ratessms = [], []
        dindex = dict()
        for i in range(0, len(cs)):
            curc = cs[i]
            if curc != -1:
                dindex.setdefault(curc, 0)
                curidx = dindex[curc]
                # print(f"curc: {curc}, curidx: {curidx}, lenn: {len(deachskill[curc])}, currate: {deachskill[curc][curidx]}")
                correct_rate = deachskill[curc][curidx][1]
                assert curidx == deachskill[curc][curidx][0]
                cursm = 1 if correct_rate != -1 else 0
                ratessms.append(cursm)
                allrates.append(correct_rate) # 只计算当前知识点对应的未来准确率
                dindex[curc] += 1
            else:
                ratessms.append(0)
                allrates.append(0)
        # allrates, ratessms = torch.FloatTensor(allrates), torch.LongTensor(ratessms)[1:]
        
        # print(f"allrates: {allrates}, -1 num: {allrates[allrates==-1].shape}, deach len: {len(deachskill)}")
        # assert False
        return allrates, ratessms[1:]

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.

        Returns: 
            (tuple): tuple containing

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": [], "is_repeat": [], "oriqs": [], "orics": [], "orisms": [], "futurerates": [], "futuresms": []}

        allcs = set()
        for q in self.dq2c:
            allcs |= self.dq2c[q]
        numc = len(allcs) 
        print(f"numc: {numc}")
        df = pd.read_csv(sequence_path)#[0:1000]
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
            if "is_repeat" in row:
                dori["is_repeat"].append([int(_) for _ in row["is_repeat"].split(",")])

            curoqs = [int(_) for _ in row["questions"].split(",")]
            curocs = [int(_) for _ in row["concepts"].split(",")]
            curors = [int(_) for _ in row["responses"].split(",")]
            is_repeat = [int(_) for _ in row["is_repeat"].split(",")]

            # 计算未来准确率
            futurerates, futuresms = self.__generate_future__(curocs, curors, numc)
            dori["futurerates"].append(futurerates)
            dori["futuresms"].append(futuresms)
            # ccs = []
            # for q, c in zip(dori["qseqs"][-1], dori["cseqs"][-1]):
            #     if q != -1:
            #         curcs = list(self.dq2c[q]) + [-1]*(10-len(self.dq2c[q]))
            #     else:
            #         curcs = [-1]*10
            #     ccs.append(curcs)
            seqlen = len(dori["cseqs"][-1])
            cqs, ccs = [], []
            i = 0
            for q, r in zip(curoqs, is_repeat):
                if (i > 0 and r == 1) or q == -1:
                    continue
                cqs.append(q)
                i += 1
            sms = [1] * (len(cqs)-1) + [0] * (seqlen - len(cqs))
            cqs = cqs + [-1] * (seqlen - len(cqs))
            for q in cqs:
                if q != -1:
                    curcs = list(self.dq2c[q]) + [-1]*(10-len(self.dq2c[q]))
                else:
                    curcs = [-1]*10
                ccs.append(curcs)
            dori["oriqs"].append(cqs)
            dori["orics"].append(ccs)
            dori["orisms"].append(sms)
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        for key in dori:
            if key not in ["rseqs", "futurerates"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        return dori
