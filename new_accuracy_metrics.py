

def compute_recall(pars, rars, alpha, bias_function):
    if len(rars) == 0 or len(pars) == 0:
        return 0

    i=0
    j=0
    value=0
    beta = 1 - alpha

    while i < len(rars):
        overlap = 0
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
def compute_metrics(pars, rars, alpha_recall, beta_Fscore):
    

    precision = compute_precision(pars, rars, flat)
    recall = compute_recall(pars, rars, alpha_recall, flat)

    F_beta = (1+((beta_Fscore)**2)) * ((precision * recall)/(((beta_Fscore)**2 * precision) +recall)) if (precision+recall) != 0 else 0.0
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

    epsilon = 0.000000001
    
    fail_count =0
    print("Assume alpha_r=0.8")
    
    #Recall
    #Full coverage
    #Equals
    #single_b
    rars = [[5,9]]
    pars = [[5,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 1.0
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1
    print("Recall, Full coverage, Equals, single_b, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)
 
    #Recall
    #Full coverage
    #Equals
    #multiple_b
    rars = [[5,9]]
    pars = [[5,5],[6,6],[7,7],[8,8],[9,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Full coverage, Equals, multiple_b's, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)

    #Recall
    #Full coverage
    #Starts
    #single_b
    rars = [[5,9]]
    pars = [[5,10]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 1.0
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Full coverage, Starts, single_b, pars=", pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)    

    #Recall
    #Full coverage
    #Starts
    #multiple_b's
    rars = [[5,9]]
    pars = [[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Full coverage, Starts, multiple_b's, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)


    #Recall
    #Full coverage
    #During
    #single_b
    rars = [[5,9]]
    pars = [[4,10]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 1.0
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Full coverage, During, single_b, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)    

    #Recall
    #Full coverage
    #During
    #multiple_b's
    rars = [[5,9]]
    pars = [[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Full coverage, During, multiple_b's, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)

    #Recall
    #Full coverage
    #Finishes
    #single_b
    rars = [[5,9]]
    pars = [[4,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 1.0
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    

    print("Recall, Full coverage, Finishes, single_b, pars=", pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)    

    #Recall
    #Full coverage
    #Finishes
    #multiple_b's
    rars = [[5,9]]
    pars = [[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1
    print("Recall, Full coverage, Finishes, multiple_b's, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)


    #Recall
    #Partial coverage
    #Overlaps
    #single_b
    rars = [[5,9]]
    pars = [[8,12]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.88
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Overlaps, single_b, pars=", pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)    

    #Recall
    #Partial coverage
    #Overlaps
    #multiple_b's & contiguous
    rars = [[5,9]]
    pars = [[8,8],[9,9],[10,10],[11,11],[12,12]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1

    print("Recall, Partial coverage, Overlaps, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)

    #Recall
    #Partial coverage
    #Overlaps
    #multiple_b's & non-contiguous
    rars = [[5,9]]
    pars = [[6,6], [8,8], [9,9], [10,10], [11,11]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Overlaps, multiple_b's & non-contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)

    #Recall
    #Partial coverage
    #Overlapped by
    #single_b
    rars = [[13,17]]
    pars = [[8,14]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.88
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Overlapped by, single_b, pars=", pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)    

    #Recall
    #Partial coverage
    #Overlapped by
    #multiple_b's & contiguous
    rars = [[13,17]]
    pars = [[8,8],[9,9],[10,10],[11,11],[12,12], [13,13], [14,14]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Overlapped by, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)


    #Recall
    #Partial coverage
    #Overlapped by
    #multiple_b's & non-contiguous
    rars = [[13,17]]
    pars = [[8,8],[9,9],[10,10],[11,11],[12,12], [13,13],[15, 15]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1
    print("Recall, Partial coverage, Overlapped by, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)

    #Recall
    #Partial coverage
    #Started by
    #single_b
    rars = [[5,9]]
    pars = [[5,6]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.88
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Started by, single_b, pars=", pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)    

    #Recall
    #Partial coverage
    #Started by
    #multiple_b's & contiguous
    rars = [[5,9]]
    pars = [[5,5],[6,6]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Started by, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)

    #Recall
    #Partial coverage
    #Started by
    #multiple_b's & non-contiguous
    rars = [[5,9]]
    pars = [[5,5], [7,7]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Started by, multiple_b's & non-contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)


    #Recall
    #Partial coverage
    #Finished by
    #single_b
    rars = [[5,9]]
    pars = [[8,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.88
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Finished by, single_b, pars=", pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)    

    #Recall
    #Partial coverage
    #Finished by
    #multiple_b's & contiguous
    rars = [[5,9]]
    pars = [[8,8],[9,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Finished by, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)

    #Recall
    #Partial coverage
    #Finished by
    #multiple_b's & non-contiguous
    rars = [[5,9]]
    pars = [[7,7], [9,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Finished by, multiple_b's & non-contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)

    #Recall
    #Partial coverage
    #During
    #single_b
    rars = [[5,9]]
    pars = [[7,8]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.88
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, During, single_b, pars=", pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)    

    #Recall
    #Partial coverage
    #During
    #multiple_b's & contiguous
    rars = [[5,9]]
    pars = [[7,7],[8,8]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, During, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)

    #Recall
    #Partial coverage
    #Started by
    #multiple_b's & non-contiguous
    rars = [[5,9]]
    pars = [[6,6], [8,8]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    recall_expected = 0.84
    success = abs(recall_expected-recall)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, During, multiple_b's & non-contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)


    
    # #Start testing precision!!!!!!!!!!!
    


    #Precision
    #Full coverage
    #Equals
    #single_b
    pars = [[5,9]]
    rars = [[5,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 1.0
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1
    print("Precision, Full coverage, Equals, single_b, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)
 
    #Precision
    #Full coverage
    #Equals
    #multiple_b
    pars = [[5,9]]
    rars = [[5,5],[6,6],[7,7],[8,8],[9,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Full coverage, Equals, multiple_b's, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)

    #Precision
    #Full coverage
    #Starts
    #single_b
    pars = [[5,9]]
    rars = [[5,10]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 1.0
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Full coverage, Starts, single_b, pars=", pars, ", rars=",rars,"precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)    

    #Precision
    #Full coverage
    #Starts
    #multiple_b's
    pars = [[5,9]]
    rars = [[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Full coverage, Starts, multiple_b's, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)


    #Precision
    #Full coverage
    #During
    #single_b
    pars = [[5,9]]
    rars = [[4,10]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 1.0
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Full coverage, During, single_b, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)    

    #Precision
    #Full coverage
    #During
    #multiple_b's
    pars = [[5,9]]
    rars = [[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Full coverage, During, multiple_b's, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)

    #Precision
    #Full coverage
    #Finishes
    #single_b
    pars = [[5,9]]
    rars = [[4,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 1.0
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    

    print("Precision, Full coverage, Finishes, single_b, pars=", pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)    

    #Precision
    #Full coverage
    #Finishes
    #multiple_b's
    pars = [[5,9]]
    rars = [[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1
    print("Precision, Full coverage, Finishes, multiple_b's, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)


    #Precision
    #Partial coverage
    #Overlaps
    #single_b
    pars = [[5,9]]
    rars = [[8,12]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.4
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, Overlaps, single_b, pars=", pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)    

    #Precision
    #Partial coverage
    #Overlaps
    #multiple_b's & contiguous
    pars = [[5,9]]
    rars = [[8,8],[9,9],[10,10],[11,11],[12,12]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1

    print("Precision, Partial coverage, Overlaps, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)

    #Precision
    #Partial coverage
    #Overlaps
    #multiple_b's & non-contiguous
    pars = [[5,9]]
    rars = [[6,6], [8,8], [9,9], [10,10], [11,11]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, Overlaps, multiple_b's & non-contiguous, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)

    #Precision
    #Partial coverage
    #Overlapped by
    #single_b
    pars = [[13,17]]
    rars = [[8,14]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.4
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, Overlapped by, single_b, pars=", pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)    

    #Precision
    #Partial coverage
    #Overlapped by
    #multiple_b's & contiguous
    pars = [[13,17]]
    rars = [[8,8],[9,9],[10,10],[11,11],[12,12], [13,13], [14,14]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, Overlapped by, multiple_b's & contiguous, pars=",pars, ", rars=",rars,"precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)


    #Precision
    #Partial coverage
    #Overlapped by
    #multiple_b's & non-contiguous
    pars = [[13,17]]
    rars = [[8,8],[9,9],[10,10],[11,11],[12,12], [13,13],[15, 15]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1
    print("Precision, Partial coverage, Overlapped by, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)

    #Precision
    #Partial coverage
    #Started by
    #single_b
    pars = [[5,9]]
    rars = [[5,6]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.4
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, Started by, single_b, pars=", pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)    

    #Precision
    #Partial coverage
    #Started by
    #multiple_b's & contiguous
    pars = [[5,9]]
    rars = [[5,5],[6,6]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Recall, Partial coverage, Started by, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "recall_expected: ", recall_expected, ", recall_acutal: ", recall, ", precision_actual: ", precision, "success: ", success)

    #Precision
    #Partial coverage
    #Started by
    #multiple_b's & non-contiguous
    pars = [[5,9]]
    rars = [[5,5], [7,7]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, Started by, multiple_b's & non-contiguous, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)


    #Precision
    #Partial coverage
    #Finished by
    #single_b
    pars = [[5,9]]
    rars = [[8,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.4
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, Finished by, single_b, pars=", pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)    

    #Precision
    #Partial coverage
    #Finished by
    #multiple_b's & contiguous
    pars = [[5,9]]
    rars = [[8,8],[9,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, Finished by, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)

    #Precision
    #Partial coverage
    #Finished by
    #multiple_b's & non-contiguous
    pars = [[5,9]]
    rars = [[7,7], [9,9]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, Finished by, multiple_b's & non-contiguous, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)

    #Precision
    #Partial coverage
    #During
    #single_b
    pars = [[5,9]]
    rars = [[7,8]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.4
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, During, single_b, pars=", pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)    

    #Precision
    #Partial coverage
    #During
    #multiple_b's & contiguous
    pars = [[5,9]]
    rars = [[7,7],[8,8]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, During, multiple_b's & contiguous, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)

    #Precision
    #Partial coverage
    #Started by
    #multiple_b's & non-contiguous
    pars = [[5,9]]
    rars = [[6,6], [8,8]]
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    precision_expected = 0.2
    success = abs(precision_expected-precision)<epsilon
    if success==False:
        fail_count += 1    
    print("Precision, Partial coverage, During, multiple_b's & non-contiguous, pars=",pars, ", rars=",rars, "precision_expected: ", precision_expected, ", precision_acutal: ", precision, ", recall_actual: ", recall, "success: ", success)

    print("Additional Tests")
    pars = [[5,8], [13,16]]
    rars = [[7,16]]
    precision_expected = 0.75
    recall_expected = 0.86
    precision, recall, _, _ = compute_metrics(pars, rars, 0.8, 0.1)
    success = (abs(precision_expected-precision)<epsilon) and (abs(recall_expected-recall)<epsilon)
    if success==False:
        fail_count += 1    

    print("pars=",pars, ", rars=",rars,"precision_expected: ", precision_expected, ", precision_acutal: ", precision, "recall_expected: ", recall_expected, ", recall_actual: ", recall, "success: ", success)

    pars = [[4,8]]
    rars = [[0,4], [8,12]]
    precision_expected = 0.2
    recall_expected = 0.84
    precision, recall, _, f1 = compute_metrics(pars, rars, 0.8, 0.1)
    success = (abs(precision_expected-precision)<epsilon) and (abs(recall_expected-recall)<epsilon)
    if success==False:
        fail_count += 1    

    print("pars=",pars, ", rars=",rars,"precision_expected: ", precision_expected, ", precision_acutal: ", precision, "recall_expected: ", recall_expected, ", recall_actual: ", recall, "success: ", success)
    print("f1: ",f1)

    pars = [[3,7]]
    rars = [[0,4], [8,12]]
    precision_expected = 0.4
    recall_expected = 0.44
    precision, recall, _, f1 = compute_metrics(pars, rars, 0.8, 0.1)
    success = (abs(precision_expected-precision)<epsilon) and (abs(recall_expected-recall)<epsilon)
    if success==False:
        fail_count += 1    

    print("pars=",pars, ", rars=",rars,"precision_expected: ", precision_expected, ", precision_acutal: ", precision, "recall_expected: ", recall_expected, ", recall_actual: ", recall, "success: ", success)
    print("f1: ",f1)

    print("fail_count: ", fail_count)
    # pars = [[7,11]]
    # rars = [[6,7]]
    # precision, recall, F_beta, F1 = compute_metrics(pars, rars, 0.8, 0.1)
    # print("precsion: ", precision, ", recall: ", recall)

    

if __name__ == '__main__':
    main()
