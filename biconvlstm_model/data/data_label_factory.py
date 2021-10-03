def __FD_labeler(data_files):
    file_names = [name.split('/')[-1] for name in data_files]
    FD_file_names = []
    for one_file in file_names : 
      if one_file.split('-')[2] == 'FD':
        FD_file_names.append(one_file)
    labels = [1 if name[3]=='V' else 0 for name in FD_file_names]
    return list(zip(data_files, labels))

def __RWF_labeler(data_files):
    file_names = [name.split('/')[-1] for name in data_files]
    RWF_file_names = []
    for one_file in file_names : 
      if one_file.split('-')[2] == 'RWF':
        RWF_file_names.append(one_file)
    labels = [1 if name[3]=='V' else 0 for name in RWF_file_names]
    return list(zip(data_files, labels))

def __UCF_labeler(data_files):
    file_names = [name.split('/')[-1] for name in data_files]
    UCF_file_names = []
    for one_file in file_names :
      if one_file.split('-')[2] == 'UCF':
        UCF_file_names.append(one_file)
    labels = [1 if name[3]=='V' else 0 for name in UCF_file_names]
    return list(zip(data_files, labels))

def __AH_labeler(data_files):
    file_names = [name.split('/')[-1] for name in data_files]
    AH_file_names = []
    for one_file in file_names : 
      if one_file.split('-')[2] == 'AH':
        AH_file_names.append(one_file)
    labels = [1 if name[3]=='V' else 0 for name in AH_file_names]
    return list(zip(data_files, labels))

def __YT_labeler(data_files):
    file_names = [name.split('/')[-1] for name in data_files]
    labels = [1 if name[3]=='V' else 0 for name in file_names]
    return list(zip(data_files, labels))


def __ALL_labeler(data_files):
    file_names = [name.split('/')[-1] for name in data_files]
    ALL_file_names = []
    for one_file in file_names :
        ALL_file_names.append(one_file)
    labels = [1 if name[3]=='V' else 0 for name in ALL_file_names]
    return list(zip(data_files, labels))

def label_factory(data_name):
    if data_name == 'FD': return __FD_labeler
    if data_name == 'AH': return __AH_labeler
    if data_name == 'YT': return __YT_labeler
    if data_name == 'RWF': return __RWF_labeler
    if data_name == 'UCF': return __UCF_labeler
    if data_name == 'ALL': return __ALL_labeler
    assert 0, "Bad data name: " + data_name
