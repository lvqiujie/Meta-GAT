import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import random
import numpy as np



class MyDataset(data.Dataset):
    def __init__(self, smiles_tasks_csv, k_shot, k_spt_pos, k_spt_neg, k_query, tasks, task_num, batchsz, type):
        super(MyDataset, self).__init__()
        self.smiles_tasks_csv = smiles_tasks_csv

        self.k_shot = k_shot  # k-shot
        self.k_spt_pos = k_spt_pos  # k-shot positive
        self.k_spt_neg = k_spt_neg  # k-shot negative
        self.k_query = k_query  # for evaluation
        self.tasks = tasks
        self.task_num = task_num
        self.batchsz = batchsz
        self.all_smi = np.array(list(self.smiles_tasks_csv["remained_smiles"].values))

        self.batchs_data = []
        self.create_batch2()

    def select_query(self, cls_label, index_list,):
        test_idx = np.random.choice(len(index_list), self.k_query, False)
        np.random.shuffle(test_idx)
        query_x = list(self.all_smi[index_list[test_idx]])
        query_y = list(cls_label[index_list[test_idx]])

        # query_x = np.array(query_x)
        query_y = np.array(query_y)

        return query_x, query_y


    # def create_test(self):
    #
    #     for b in range(self.batchsz):  # for each batch
    #         # 1.select n_way classes randomly
    #         # selected_cls = random.choices(self.tasks, k=self.task_num)  #duplicate
    #         selected_cls = random.sample(self.tasks, k=self.task_num)  # no duplicate
    #         np.random.shuffle(selected_cls)
    #         support_x = []
    #         support_y = []
    #         query_x = []
    #         query_y = []
    #         for cls in selected_cls:
    #             # 2. select k_shot + k_query for each class
    #             cls_label = np.array(self.smiles_tasks_csv[cls])
    #             test_index = np.where(cls_label == 0 | cls_label == 1)[0]
    #             test_idx = np.random.choice(len(test_index), self.k_shot + self.k_query, False)
    #             np.random.shuffle(test_idx)
    #             test_all = list(self.all_smi[test_index[test_idx]])
    #             test_label_all = list(cls_label[test_index[test_idx]])
    #             support_x.append(test_all[:self.k_shot])
    #             query_x.append(test_all[self.k_shot:])
    #             support_y.append(test_label_all[:self.k_shot])
    #             query_y.append(test_label_all[self.k_shot:])
    #
    #
    #         # support_x = np.array(support_x)
    #         support_y = np.array(support_y)
    #
    #
    #         # query_x = np.array(query_x)
    #         query_y = np.array(query_y)
    #
    #         self.batchs_data.append([support_x, support_y, query_x, query_y])
    #
    # def create_batch(self):
    #     """
    #     create batch for meta-learning.
    #     ×episode× here means batch, and it means how many sets we want to retain.
    #     :param episodes: batch size
    #     :return:
    #     """
    #     for b in range(self.batchsz):  # for each batch
    #         # 1.select n_way classes randomly
    #         # selected_cls = random.choices(self.tasks, k=self.task_num)  #duplicate
    #         selected_cls = random.sample(self.tasks, k=self.task_num)  # no duplicate
    #         np.random.shuffle(selected_cls)
    #         support_x = []
    #         support_y = []
    #         query_x = []
    #         query_y = []
    #         for cls in selected_cls:
    #             # 2. select k_shot + k_query for each class
    #             cls_label = np.array(self.smiles_tasks_csv[cls])
    #             if random.random() > 0.5:
    #                 negative_index = np.where(cls_label == 0)[0]
    #                 negative_idx = np.random.choice(len(negative_index), self.k_shot + self.k_query, False)
    #                 np.random.shuffle(negative_idx)
    #                 negative_all = list(self.all_smi[negative_index[negative_idx]])
    #                 negative_label_all = list(cls_label[negative_index[negative_idx]])
    #                 support_x.append(negative_all[:self.k_shot])
    #                 query_x.append(negative_all[self.k_shot:])
    #                 support_y.append(negative_label_all[:self.k_shot])
    #                 query_y.append(negative_label_all[self.k_shot:])
    #             else:
    #                 positive_index = np.where(cls_label == 1)[0]
    #                 positive_idx = np.random.choice(len(positive_index), self.k_shot + self.k_query, False)
    #                 np.random.shuffle(positive_idx)
    #                 positive_all = list(self.all_smi[positive_index[positive_idx]])
    #                 positive_label_all = list(cls_label[positive_index[positive_idx]])
    #                 support_x.append(positive_all[:self.k_shot])
    #                 query_x.append(positive_all[self.k_shot:])
    #                 support_y.append(positive_label_all[:self.k_shot])
    #                 query_y.append(positive_label_all[self.k_shot:])
    #
    #         # support_x = np.array(support_x)
    #         support_y = np.array(support_y)
    #
    #
    #         # query_x = np.array(query_x)
    #         query_y = np.array(query_y)
    #
    #         self.batchs_data.append([support_x, support_y, query_x, query_y])
    #
    def create_batch2(self):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        for b in range(self.batchsz):  # for each batch
            # 1.select n_way classes randomly
            # selected_cls = random.choices(self.tasks, k=self.task_num)  #duplicate
            selected_cls = random.sample(self.tasks, k=self.task_num)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                cls_label = np.array(self.smiles_tasks_csv[cls])
                cls_support_x = []
                cls_support_y = []
                cls_query_x = []
                cls_query_y = []
                negative_index = np.where(cls_label == 0)[0]
                negative_idx = np.random.choice(len(negative_index), self.k_spt_neg + self.k_query //2, False)
                np.random.shuffle(negative_idx)
                negative_all = list(self.all_smi[negative_index[negative_idx]])
                negative_label_all = list(cls_label[negative_index[negative_idx]])
                cls_support_x.extend(negative_all[:self.k_spt_neg])
                cls_support_y.extend(negative_label_all[:self.k_spt_neg])
                cls_query_x.extend(negative_all[self.k_spt_neg:])
                cls_query_y.extend(negative_label_all[self.k_spt_neg:])


                positive_index = np.where(cls_label == 1)[0]
                positive_idx = np.random.choice(len(positive_index), self.k_spt_pos + self.k_query //2, False)
                np.random.shuffle(positive_idx)
                positive_all = list(self.all_smi[positive_index[positive_idx]])
                positive_label_all = list(cls_label[positive_index[positive_idx]])
                cls_support_x.extend(positive_all[:self.k_spt_pos])
                cls_support_y.extend(positive_label_all[:self.k_spt_pos])
                cls_query_x.extend(positive_all[self.k_spt_pos:])
                cls_query_y.extend(positive_label_all[self.k_spt_pos:])

                c = list(zip(cls_query_x, cls_query_y))
                random.shuffle(c)
                cls_query_x[:], cls_query_y[:] = zip(*c)

                support_x.append(cls_support_x)
                support_y.append(cls_support_y)
                query_x.append(cls_query_x)
                query_y.append(cls_query_y)
            # support_x = np.array(support_x)
            support_y = np.array(support_y)


            # query_x = np.array(query_x)
            query_y = np.array(query_y)

            self.batchs_data.append([support_x, support_y, query_x, query_y])

    def create_batch3(self):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        for b in range(self.batchsz):  # for each batch
            # 1.select n_way classes randomly
            # selected_cls = random.choices(self.tasks, k=self.task_num)  #duplicate
            selected_cls = random.sample(self.tasks, k=self.task_num)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                cls_label = np.array(self.smiles_tasks_csv[cls])
                cls_support_x = []
                cls_support_y = []

                negative_index = np.where(cls_label == 0)[0]
                negative_idx = np.random.choice(len(negative_index), self.k_spt_neg, False)
                np.random.shuffle(negative_idx)
                negative_all = list(self.all_smi[negative_index[negative_idx]])
                negative_label_all = list(cls_label[negative_index[negative_idx]])
                cls_support_x.extend(negative_all)
                cls_support_y.extend(negative_label_all)
                # cls_query_x.extend(negative_all[query_negative_num:])
                # cls_query_y.extend(negative_label_all[query_negative_num:])


                positive_index = np.where(cls_label == 1)[0]
                positive_idx = np.random.choice(len(positive_index), self.k_spt_pos, False)
                np.random.shuffle(positive_idx)
                positive_all = list(self.all_smi[positive_index[positive_idx]])
                positive_label_all = list(cls_label[positive_index[positive_idx]])
                cls_support_x.extend(positive_all)
                cls_support_y.extend(positive_label_all)
                # cls_query_x.extend(positive_all[support_positive_num:])
                # cls_query_y.extend(positive_label_all[support_positive_num:])

                all_index = np.where((cls_label == 0) | (cls_label == 1))[0]
                all_index = [i for i in all_index if i not in negative_idx]
                all_index = [i for i in all_index if i not in positive_idx]
                # all_index.remove(negative_idx)
                # all_index.drop(positive_idx)
                cls_query_x, cls_query_y = self.select_query(cls_label, np.array(all_index))

                c = list(zip(cls_query_x, cls_query_y))
                random.shuffle(c)
                cls_query_x[:], cls_query_y[:] = zip(*c)

                c = list(zip(cls_support_x, cls_support_y))
                random.shuffle(c)
                cls_support_x[:], cls_support_y[:] = zip(*c)

                support_x.append(cls_support_x)
                support_y.append(cls_support_y)
                query_x.append(cls_query_x)
                query_y.append(cls_query_y)


            # support_x = np.array(support_x)
            support_y = np.array(support_y)


            # query_x = np.array(query_x)
            query_y = np.array(query_y)

            self.batchs_data.append([support_x, support_y, query_x, query_y])

    def __getitem__(self, item):
        x_spt, y_spt, x_qry, y_qry = self.batchs_data[item]
        return x_spt, y_spt, x_qry, y_qry

    def __len__(self):
        return len(self.batchs_data)