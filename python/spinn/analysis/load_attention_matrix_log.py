



WAIT_ID = 1
WAIT_SIZE = 2
WAIT_MAT = 3

def load_attention_matrix(file_path):

    attention_matrix = {}
    stage = WAIT_ID   # 1: wait for sentence id, 2: wait for size, 3: wait for attention matrix
    one = {}
    count = 0
    with open(file_path, 'r') as txtfile:
        for row in txtfile:
            if stage == WAIT_ID:
                one = {'id': row[:-1], 'matrix': []}    # -1 is to escape the \n
                stage = WAIT_SIZE
            elif stage == WAIT_SIZE:
                (hsize, psize) = [int(x) for x in row.split(',')]
                one['hsize'] = hsize
                one['psize'] = psize
                stage = WAIT_MAT
                count = hsize
            else :
                one['matrix'].append([float(x) for x in row.split(',')]) # read a line, convert to float
                count -= 1
                if count == 0:
                    stage = WAIT_ID
                    attention_matrix[one['id']] = one

    return attention_matrix



if __name__ == '__main__':

    # demo
    atts = load_attention_matrix('/Users/Alex/mlworkspace/spinn/debug_att/snli-ATTSPINN-1491099189/attention-matrix-10.txt')

    print len(atts)


