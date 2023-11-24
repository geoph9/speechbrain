"""
AMI corpus contained 100 hours of meeting recording.
This script returns the standard train, dev and eval split for AMI corpus.
For more information on dataset please refer to http://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml
"""

ALLOWED_OPTIONS = ["scenario_only", "full_corpus", "full_corpus_asr"]

MEETINGS = {
    'EN2001': ['EN2001a', 'EN2001b', 'EN2001d', 'EN2001e'],
    'EN2002': ['EN2002a', 'EN2002b', 'EN2002c', 'EN2002d'],
    'EN2003': ['EN2003a'],
    'EN2004': ['EN2004a'],
    'EN2005': ['EN2005a'],
    'EN2006': ['EN2006a','EN2006b'],
    'EN2009': ['EN2009b','EN2009c','EN2009d'],
    'ES2002': ['ES2002a','ES2002b','ES2002c','ES2002d'],
    'ES2003': ['ES2003a','ES2003b','ES2003c','ES2003d'],
    'ES2004': ['ES2004a','ES2004b','ES2004c','ES2004d'],
    'ES2005': ['ES2005a','ES2005b','ES2005c','ES2005d'],
    'ES2006': ['ES2006a','ES2006b','ES2006c','ES2006d'],
    'ES2007': ['ES2007a','ES2007b','ES2007c','ES2007d'],
    'ES2008': ['ES2008a','ES2008b','ES2008c','ES2008d'],
    'ES2009': ['ES2009a','ES2009b','ES2009c','ES2009d'],
    'ES2010': ['ES2010a','ES2010b','ES2010c','ES2010d'],
    'ES2011': ['ES2011a','ES2011b','ES2011c','ES2011d'],
    'ES2012': ['ES2012a','ES2012b','ES2012c','ES2012d'],
    'ES2013': ['ES2013a','ES2013b','ES2013c','ES2013d'],
    'ES2014': ['ES2014a','ES2014b','ES2014c','ES2014d'],
    'ES2015': ['ES2015a','ES2015b','ES2015c','ES2015d'],
    'ES2016': ['ES2016a','ES2016b','ES2016c','ES2016d'],
    'IB4001': ['IB4001'],
    'IB4002': ['IB4002'],
    'IB4003': ['IB4003'],
    'IB4004': ['IB4004'],
    'IB4005': ['IB4005'],
    'IB4010': ['IB4010'],
    'IB4011': ['IB4011'],
    'IN1001': ['IN1001'],
    'IN1002': ['IN1002'],
    'IN1005': ['IN1005'],
    'IN1007': ['IN1007'],
    'IN1008': ['IN1008'],
    'IN1009': ['IN1009'],
    'IN1012': ['IN1012'],
    'IN1013': ['IN1013'],
    'IN1014': ['IN1014'],
    'IN1016': ['IN1016'],
    'IS1000': ['IS1000a','IS1000b','IS1000c','IS1000d'],
    'IS1001': ['IS1001a','IS1001b','IS1001c','IS1001d'],
    'IS1002': ['IS1002b','IS1002c','IS1002d'],
    'IS1003': ['IS1003a','IS1003b','IS1003c','IS1003d'],
    'IS1004': ['IS1004a','IS1004b','IS1004c','IS1004d'],
    'IS1005': ['IS1005a','IS1005b','IS1005c'],
    'IS1006': ['IS1006a','IS1006b','IS1006c','IS1006d'],
    'IS1007': ['IS1007a','IS1007b','IS1007c','IS1007d'],
    'IS1008': ['IS1008a','IS1008b','IS1008c','IS1008d'],
    'IS1009': ['IS1009a','IS1009b','IS1009c','IS1009d'],
    'TS3003': ['TS3003a','TS3003b','TS3003c','TS3003d'],
    'TS3004': ['TS3004a','TS3004b','TS3004c','TS3004d'],
    'TS3005': ['TS3005a','TS3005b','TS3005c','TS3005d'],
    'TS3006': ['TS3006a','TS3006b','TS3006c','TS3006d'],
    'TS3007': ['TS3007a','TS3007b','TS3007c','TS3007d'],
    'TS3008': ['TS3008a','TS3008b','TS3008c','TS3008d'],
    'TS3009': ['TS3009a','TS3009b','TS3009c','TS3009d'],
    'TS3010': ['TS3010a','TS3010b','TS3010c','TS3010d'],
    'TS3011': ['TS3011a','TS3011b','TS3011c','TS3011d'],
    'TS3012': ['TS3012a','TS3012b','TS3012c','TS3012d'],
}

PARTITIONS = {
    'scenario-only': {
        'train': [meeting for session in [
                'ES2002','ES2005','ES2006','ES2007','ES2008','ES2009','ES2010','ES2012','ES2013',
                'ES2015','ES2016','IS1000','IS1001','IS1002','IS1003','IS1004','IS1005','IS1006',
                'IS1007','TS3005','TS3008','TS3009','TS3010','TS3011','TS3012'
            ] for meeting in MEETINGS[session] if meeting not in ['IS1002a','IS1005d']],
        'dev': [meeting for session in [
                'ES2003','ES2011','IS1008','TS3004','TS3006'
            ] for meeting in MEETINGS[session]],
        'test': [meeting for session in [
                'ES2004','ES2014','IS1009','TS3003','TS3007'
            ] for meeting in MEETINGS[session]]
    },
    'full_corpus': {
        'train': [meeting for session in [
                'ES2002','ES2005','ES2006','ES2007','ES2008','ES2009','ES2010','ES2012','ES2013',
                'ES2015','ES2016','IS1000','IS1001','IS1002','IS1003','IS1004','IS1005','IS1006',
                'IS1007','TS3005','TS3008','TS3009','TS3010','TS3011','TS3012','EN2001','EN2003',
                'EN2004','EN2005','EN2006','EN2009','IN1001','IN1002','IN1005','IN1007','IN1008',
                'IN1009','IN1012','IN1013','IN1014','IN1016'
            ] for meeting in MEETINGS[session]],
        'dev': [meeting for session in [
                'ES2003','ES2011','IS1008','TS3004','TS3006','IB4001','IB4002','IB4003','IB4004',
                'IB4010','IB4011'
            ] for meeting in MEETINGS[session]],
        'test': [meeting for session in [
                'ES2004','ES2014','IS1009','TS3003','TS3007','EN2002'
            ] for meeting in MEETINGS[session]]
    },
    'full_corpus_asr': {
        'train': [meeting for session in [
                'ES2002','ES2005','ES2006','ES2007','ES2008','ES2009','ES2010','ES2012','ES2013',
                'ES2015','ES2016','IS1000','IS1001','IS1002','IS1003','IS1004','IS1005','IS1006',
                'IS1007','TS3005','TS3008','TS3009','TS3010','TS3011','TS3012','EN2001','EN2003',
                'EN2004','EN2005','EN2006','EN2009','IN1001','IN1002','IN1005','IN1007','IN1008',
                'IN1009','IN1012','IN1013','IN1014','IN1016','ES2014','TS3007','ES2003','TS3006'
            ] for meeting in MEETINGS[session]],
        'dev': [meeting for session in [
                'ES2011','IS1008','TS3004','IB4001','IB4002','IB4003','IB4004','IB4010','IB4011'
            ] for meeting in MEETINGS[session]],
        'test': [meeting for session in [
                'ES2004','IS1009','TS3003','EN2002'
            ] for meeting in MEETINGS[session]]
    }
}

MICS = ['ihm','ihm-mix','sdm','mdm','mdm8-bf']
MDM_ARRAYS = ['Array1','Array2']
MDM_CHANNELS = ['01','02','03','04','05','06','07','08']


def get_AMI_split(split_option):
    """
    Prepares train, dev, and test sets for given split_option

    Arguments
    ---------
    split_option: str
        The standard split option.
        Allowed options: "scenario_only", "full_corpus", "full_corpus_asr"

    Returns
    -------
        Meeting IDs for train, dev, and test sets for given split_option
    """

    if split_option not in ALLOWED_OPTIONS:
        print(
            f'Invalid split "{split_option}" requested!\nValid split_options are: ',
            ALLOWED_OPTIONS,
        )
        return

    splits = PARTITIONS[split_option]
    train = splits['train']
    dev = splits['dev']
    test = splits['test']
    return train, dev, test
