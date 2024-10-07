DIR-EC
Intra-region enhancement and inter-region collaboration network for facial expression recognition


To suppress the influence of occlusions and posture variations on facial expression recognition (FER) in natural scenes, a facial expression recognition network based on intra-region enhancement and inter-region collaboration (DIR-EC) is proposed. The proposed network mainly contains intra-regional multi-scale enhancement subnet (Intra-MSEnet), inter-regional multi-granularity collaborative subnet (Inter-MGCnet), and adaptive fusion subnet (AFSnet). In the Intra-MSEnet, a down-up dual attention mechanism (DUAM) is constructed to extract the intra-regional low-level spatial semantics from up to down and high-level channel semantics from down to up. In the Inter-MGCnet, a collaborative guidance attention structure (CGAS) is designed to capture inter-regional multi-granularity collaborative semantics of local features and global features, which achieves the guidance of coarse-grained features to fine-grained features. In the AFSnet, an adaptive fusion strategy is proposed to fuse inter-regional collaborative semantics and global guidance semantics. The experimental results show that the expression recognition rates of DIR-EC are 90.14% and 90.32% on RAF-DB and FERPlus datasets, which are 13.71% and 11.01% higher than the baseline method, respectively. Compared with related methods, the proposed DIR-EC improves the expression recognition performance in natural scenes, and reduces the influence of occlusions and posture variations.
Dataset

Download RAF-DB and FER-FERPlus datasets.

URLs: RAF-DB: http://www.whdeng.cn/raf/model1.html.  FERPlus: https://github.com/Microsoft/FERPlus.
