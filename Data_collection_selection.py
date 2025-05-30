import pandas as pd

def load_dataset():
  column_names = [
      "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
      "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
      "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt",
      "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
      "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
      "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"
  ]

  df1 = pd.read_csv("/DS_term/UNSW-NB15_1.csv", header=None, names=column_names, low_memory=False)
  df2 = pd.read_csv("/DS_term/UNSW-NB15_2.csv", header=None, names=column_names, low_memory=False)
  df3 = pd.read_csv("/DS_term/UNSW-NB15_3.csv", header=None, names=column_names, low_memory=False)
  df4 = pd.read_csv("/DS_term/UNSW-NB15_4.csv", header=None, names=column_names, low_memory=False)

  df1_0 = df1[df1['Label'] == 0]
  df2_0 = df2[df2['Label'] == 0]
  df3_0 = df3[df3['Label'] == 0]
  df4_0 = df4[df4['Label'] == 0]

  df1_1 = df1[df1['Label'] == 1]
  df2_1 = df2[df2['Label'] == 1]
  df3_1 = df3[df3['Label'] == 1]
  df4_1 = df4[df4['Label'] == 1]

  df_label0 = pd.concat([df1_0, df2_0, df3_0, df4_0], ignore_index=True)
  df_label1 = pd.concat([df1_1, df2_1, df3_1, df4_1], ignore_index=True)


  df_label1 = df_label1[df_label1['attack_cat'].isin(["Generic", "Reconnaissance"])]
  df_label0 = df_label0.sample(frac=0.25)

  df_final = pd.concat([df_label0, df_label1])
  df_final = df_final.drop('attack_cat', axis=1)

  df_final.to_csv("/DS_term/1_Raw_DataSet.csv", index=False)

if __name__ == '__main__':
  load_dataset()
