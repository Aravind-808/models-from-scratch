import torch

def ctoi(char_array):
    char_to_int = {char: int+1 for (int, char) in enumerate(char_array)}
    char_to_int['<>'] = 0 
    return char_to_int

def itoc(char_to_int):
    int_to_char = {index: char for (char, index) in char_to_int.items()}
    return int_to_char

def init_tensor_array(names):
    N = torch.zeros((27,27), dtype=torch.int32)
    char_array = sorted(list(set(''.join(names))))

    return N, char_array

def init_charint_map(char_array):
    char_to_int = ctoi(char_array)
    int_to_char = itoc(char_to_int)

    return char_to_int, int_to_char