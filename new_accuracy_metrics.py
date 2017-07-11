

def compute_recall(pars, rars, alpha, bias_function):
    if len(rars) == 0 or len(pars) == 0:
        return 0

    i=0
    j=0
    value=0
    beta = 1 - alpha

    while i < len(rars):
        #finding out the first par which overlaps current rar
        while (j < len(pars)) and pars[j][0] <= rars[i][1]:
            if pars[j][1] >= rars[i][0]:
                break
            j += 1
        
        #j out bounded which means no overlap, and we already finished traversing regions_2, just end the loop
        if j == len(pars):
            break

        #no overlap for current i, try next i
        if pars[j][0] > rars[i][1]:
            i += 1
            continue
        # number of distinct par's the current rar overlaps
        count = 0
         
        temp_value = 0
        while True:
            count += 1
            end = min(pars[j][1], rars[i][1])
            start = max(pars[j][0], rars[i][0])
            overlap += (end - start + 1)
            overlapping_points = set()
            for x in range(start, end+1):
                overlapping_points.add(x)

            temp_value += alpha + beta * f(rars[i], overlapping_points, bias_function)

            if (j == len(pars) - 1) or (pars[j + 1][0] > rars[i][1]):
                break
            j += 1

        #calculate the metric
        value += (temp_value/count)
        i += 1

    return value / len(rars)






def compute_precision(pars, rars, bias_function):
    if len(rars) == 0 or len(pars) == 0:
        return 0

    i=0
    j=0
    value=0

    while i < len(pars):
        overlap = 0
        #finding out the first rar which overlaps current par
        while (j < len(rars)) and rars[j][0] <= pars[i][1]:
            if rars[j][1] >= pars[i][0]:
                break
            j += 1
        
        #j has exceeded the last rar, which means there is no overlap between the current par and rars, just end the loop
        if j == len(rars):
            break

        #no overlap for current i, try next i
        if rars[j][0] > pars[i][1]:
            i += 1
            continue

        # number of distinct rar's the current par overlaps
        count = 0
         
        temp_value = 0
        while True:
            count += 1
            end = min(rars[j][1], pars[i][1])
            start = max(rars[j][0], pars[i][0])
          
            overlap += (end - start + 1)
            overlapping_points = set()
            for x in range(start, end+1):
                print(x, "\n")
                overlapping_points.add(x)

            temp_value += f(pars[i], overlapping_points, bias_function)

            if (j == len(rars) - 1) or (rars[j + 1][0] > pars[i][1]):
                break
            j += 1

        #calculate the metric
        value += (temp_value/count)
        i += 1

    return value / len(pars)

#previously known as gamma function
def f(base_range, overlap_set, bias_function):

    my_value = 0
    best = 0
    start = base_range[0]
    end = base_range[1]
    for i in range(0, end-start+1):
        bias = bias_function(len(base_range), i)
        best += bias
        if (i+start) in overlap_set:
            my_value += bias

    return my_value*1.0 / best

#compute metrics: recall, precision, F scores
#predicted_data and actual_data are both arrays of windows i.e. ([[start_1, end_1],[start_2,end_1]])
def compute_metrics(pars, rars, beta):
    alpha = 0.8

    precision = compute_precision(pars, rars, flat)
    recall = compute_recall(pars, rars, alpha, flat)

    F_beta = (1+((beta)**2)) * ((precision * recall)/(((beta)**2 * precision) +recall)) if (precision+recall) != 0 else 0.0
    F1 = 2 * ((precision * recall)/(precision + recall)) if (precision+recall) != 0 else 0.0

    return precision, recall, F_beta, F1


def flat(length, cur_index):
    return 1.0

def front_side_weighted(length, cur_index):
  return length - cur_index;


def back_side_weighted(length, cur_index):
    return cur_index+1;

def middle_weighted(length, cur_index):
    distance = 0.0;
    if index <= actual.size() / 2:
        return cur_index;
    else:
        return length - index;
  
def main():
    # if len(sys.argv) < 3:
    #     print("Usage: lstm_ad.py <train-file> <test-file>")
    #     print("Try: lstm_ad.py ss-train-r2.csv ss-test-1.csv")
    #     return

    # train_file = sys.argv[1]
    # threshold_train_file = sys.argv[2]
    # test_file = sys.argv[3]

    pars = [[7,11]]
    rars = [[6,7]]
    precision, recall, F_beta, F1 = compute_metrics(pars, rars, 0.1)
    print("precsion: ", precision, ", recall: ", recall)

    

if __name__ == '__main__':
    main()
