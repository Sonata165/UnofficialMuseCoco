## MuseCoco Input Details
Here are possible input tokens to MuseCoco
    "I1s2": "Instrument", a 28-dim multi-hot vector. ("乐器个列表，每个列表长度为3，依次为是、否、NA")
        Format: I1s2_[inst_id]_0  
        inst_id: [0-27]
        There are condition tokens like I1s2_[inst_id]_1 and ..._2. Deprecated for simplicity.
    "R1": "Rhythm Danceability", 
        R1_0: dancable
        R1_1: not dancable
        R1_2: NA
    "R3": "Rhythm Intensity",  
        R3_0: not intense
        R3_1: med
        R3_2: intense
    "S2s1": "Artist", 
        Format: S2s1_[id]
        id's value:
            'beethoven': 0,
            'mozart': 1,
            'chopin': 2,
            'schubert': 3,
            'schumann': 4,
            'bach-js': 5,
            'haydn': 6,
            'brahms': 7,
            'Handel': 8,
            'tchaikovsky': 9,
            'mendelssohn': 10,
            'dvorak': 11,
            'liszt': 12,
            'stravinsky': 13,
            'mahler': 14,
            'prokofiev': 15,
            'shostakovich': 16,
    "S4": "Genre",
        S4_[gid]_0
        gid's value:
            'New Age': 0,
            'Electronic': 1,
            'Rap': 2,
            'Religious': 3,
            'International': 4,
            'Easy_Listening': 5,
            'Avant_Garde': 6,
            'RnB': 7,
            'Latin': 8,
            'Children': 9,
            'Jazz': 10,
            'Classical': 11,
            'Comedy_Spoken': 12,
            'Pop_Rock': 13,
            'Reggae': 14,
            'Stage': 15,
            'Folk': 16,
            'Blues': 17,
            'Vocal': 18,
            'Holiday': 19,
            'Country': 20,
            'Symphony': 21,
        Similarly, S4_[gid]_1 and S4_[gid]_2 are deprecated.
    "B1s1": "Bar", represent bar个数区间的id
        B1s1_[bid]
        bid's value:
            0：1-4，
            1：5-8，
            2：9-12，
            3：13-16
    "TS1s1": "Time Signature",
        TS1s1_[tsid]
        tsid's value:
            0: (4, 4), 
            1: (2, 4), 
            2: (3, 4), 
            3: (1, 4), 
            4: (6, 8), 
            5: (3, 8)
    "K1": "Key",
        K1_0: major
        K1_1: minor
        K1_2: unknown
    "T1s1": "Tempo",
        T1s1_[tid]
        tid's value:
            0表示慢，
            1表示适中
            2表示快。
    "P4": "Pitch Range", n_octaves
        P4_[0-12]
        0个8度，1个8度，...，11个8度, NA
    "EM1": "Emotion", but don't know the mapping. Detail not specified. Deprecated.
        EM1_[0-4] 
    "TM1": “Time", output duration in seconds (Deprecated)
        TM1_[0-5]
            0表示(0-15]秒，
            1表示(15-30]秒，
            2表示30-45秒，
            3表示45-60秒，
            4表示60秒以上
