
benchmarks = {
     'IEMOCAP-DA':{
        'labels':['oth', 'ap', 'o', 'g', 's', 'a', 'b', 'c', 'ans', 'q', 'ag', 'dag'],
        'max_seq_lengths': {
                'text': 44,
                'video': 230,
                'audio': 380
            },
            'feat_dims': {
                'text': 768,
                'video': 1024,
                'audio': 768
            },
    },
    'MIntRec':{
        'labels': [
                    'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
                    'Agree', 'Taunt', 'Flaunt', 
                    'Joke', 'Oppose', 
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
                    'Prevent', 'Greet', 'Ask for help' 
        ],
        'max_seq_lengths': {
            'text': 30,
            'video': 230, 
            'audio': 480
        },
        'feat_dims': {
            'text': 768,
            'video': 1024,
            'audio': 768
        },

    },
    'MELD-DA':{
        'labels': ['a', 'ag', 'ans', 'ap', 'b', 'c', 'dag', 'g', 'o',  'q', 's', 'oth'],
        'max_seq_lengths': {
            'text': 70, 
            'video': 250,
            'audio': 520
        },
        'feat_dims': {
            'text': 768,
            'video': 1024,
            'audio': 768
        },
    },
}
