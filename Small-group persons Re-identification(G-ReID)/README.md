# Group Re-Identification Based on single feature attention learning network(SFALN)
**Abstract:** People often go together in groups, and group re-identification (G-Reid) is an important but less researched topic. The goal of G-Reid is to find a group of people under different surveillance camera perspectives. It not only faces the same challenge with traditional ReID, but also involves the changes in group layout and membership. To solve these problems, we propose a Single Feature Attention Learning Network (SFALN). The proposed network makes use of the abundant ReID datasets by transfer learning, and extracts effective feature information of the groups through attention mechanism. Experimental results on the public dataset demonstrate the state-of-the-art effectiveness of our approach.

## Framework
In the G-Reid task, we have a Query image p containing N people. Our goal is to find the corresponding group image p in the gallery set $G ={g_t}$, where t represents the $t^{th}$ group image in gallery G, let $g_{t}^{j}$represent the j^{th} person in the image g_{t}.  Our proposed framework consists of two main parts. First, the ReID-style training set is transferred to the G-Reid style using the style transfer method. Then, we send the images to the single feature attention learning network (SFALN) to extract features. During testing, these features will be used to calculate similarity of group images. The overall architecture of our proposed single feature attention learning network is shown in Fig.1. 
![framework](C:\Users\OMEN\Desktop\大学\PRCV发表\model_final_version.jpg  "Framework")
![PAM](C:\Users\OMEN\Desktop\大学\PRCV发表\pamf.jpg  "Framework")

## Implement
  - Use CycleGAN to transfer the style of market1501 into group-style dataset.
  - The transferred data should be saved in **single/data/market1501** 
  - Single representation network can be implemented as:  
    `python3 main.py -d market1501 --logs-dir logs/market1501 --epoch 30`

