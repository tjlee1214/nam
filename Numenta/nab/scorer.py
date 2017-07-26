# ----------------------------------------------------------------------
# Copyright (C) 2014-2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import math
import os
import pandas

from nab.util import (convertResultsPathToDataPath,
                      convertAnomalyScoresToDetections,
                      getProbationPeriod)



def compute_recall(pars, rars, alpha, bias_function):
  if len(rars) == 0 or len(pars) == 0:
    return 0, 0

   
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
  
  return value, value/len(rars)






def compute_precision(pars, rars, bias_function):
    if len(rars) == 0 or len(pars) == 0:
        return 0, 0

    i=0
    j=0
    value=0

    while i < len(pars):
        overlap = 0
        #find out the first rar which overlaps current par
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

    return value, value/len(pars)

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
#anomaly_flag[i] == '1' or 
#Compute an array of anomaly windows(i.e a sequence of 1's) from an array of 1's and 0's
def compute_anomaly_windows(anomaly_flag):
  anomaly_windows = []
  startIndex = -1
  endIndex = -1
  for i in range(len(anomaly_flag)):
    if anomaly_flag[i] == 1:
      if startIndex==-1:
        startIndex = i
        endIndex = i
      elif endIndex+1 != i:
        anomaly_windows.append([startIndex,endIndex])
        startIndex = i
        endIndex = i
      else:
        endIndex = i
  
  if startIndex!=-1 and (len(anomaly_windows)==0 or anomaly_windows[len(anomaly_windows)-1][0]!=startIndex):
    anomaly_windows.append([startIndex, endIndex])

  return anomaly_windows

def getRanges(self):
  return compute_anomaly_windows(self.data['label'].values.tolist())

class Window(object):
  """Class to store a single window in a datafile."""

  def __init__(self, windowId, limits, data):
    """
    @param windowId   (int)           An integer id for the window.

    @limits           (tuple)         (start timestamp, end timestamp).

    @data             (pandas.Series) Raw rows of the whole datafile.
    """
    self.id = windowId
    self.t1, self.t2 = limits

    temp = data[data["timestamp"] >= self.t1]
    self.window = temp[temp["timestamp"] <= self.t2]

    self.indices = self.window.index
    self.length = len(self.indices)


  def __repr__(self):
    """
    String representation of Window. For debugging.
    """
    s = "WINDOW id=" + str(self.id)
    s += ", limits: [" + str(self.t1) + ", " + str(self.t2) + "]"
    s += ", length: " + str(self.length)
    s += "\nwindow data:\n" + str(self.window)
    return s


  def getFirstTruePositive(self):
    """Get the index of the first true positive within a window.

    @return (int)   Index of the first occurrence of the true positive within
                    the window. -1 if there are none.
    """
    tp = self.window[self.window["type"] == "tp"]
    if len(tp) > 0:
      return tp.iloc[0].name
    else:
      return -1


class Scorer(object):
  """Class used to score a datafile."""

  def __init__(self,
               timestamps,
               predictions,
               labels,
               windowLimits,
               costMatrix,
               probationaryPeriod):
    """
    @param predictions   (pandas.Series)   Detector predictions of
                                           whether each record is anomalous or
                                           not. predictions[
                                           0:probationaryPeriod] are ignored.

    @param labels        (pandas.DataFrame) Ground truth for each record.
                                           For each record there should be a 1
                                           or a 0. A 1 implies this record is
                                           within an anomalous window.

    @param windowLimits  (list)            All the window limits in tuple
                                           form: (timestamp start, timestamp
                                           end).

    @param costmatrix    (dict)            Dictionary containing the
                                           cost matrix for this profile.
                                           type:  True positive (tp)
                                                  False positive (fp)
                                                  True Negative (tn)
                                                  False Negative (fn)

    @param probationaryPeriod
                         (int)             Row index after which predictions
                                           are scored.
    """
    self.data = pandas.DataFrame()
    self.data["timestamp"] = timestamps
    self.data["label"] = labels

    self.probationaryPeriod = probationaryPeriod
    self.costMatrix = costMatrix
    self.totalCount = len(self.data["label"])

    self.counts = {
      "tp": 0,
      "tn": 0,
      "fp": 0,
      "fn": 0}

    self.score = None
    self.length = len(predictions)
    self.data["type"] = self.getAlertTypes(predictions)
    self.windows = self.getWindows(windowLimits)


  def getWindows(self, limits):
    """Create list of windows for this datafile.

    @return (list)    All the window limits in tuple form: (timestamp start,
                      timestamp end).
    """
    # Sort windows before putting them into list
    windows = [Window(i, limit, self.data) for i, limit in enumerate(limits)]

    return windows


  def getAlertTypes(self, predictions):
    """For each record, decide whether it is a tp, fp, tn, or fn. Populate
    counts dictionary with the total number of records in each category.
    Return a list of strings containing each prediction type."""
    types = []

    for i, row in self.data.iterrows():
      if i < self.probationaryPeriod:
        types.append("probationaryPeriod")
        continue

      pred = predictions[int(i)]
      diff = abs(pred - row["label"])

      category = str()
      category += "f" if bool(diff) else "t"
      category += "p" if bool(pred) else "n"
      self.counts[category] += 1
      types.append(category)

    return types



  def getScore(self):
    """
    Score the entire datafile and return a single floating point score.
    The position in a given window is calculated as the distance from the end
    of the window, normalized [-1,0]. I.e. positions -1.0 and 0.0 are at the
    very front and back of the anomaly window, respectively.

    Flat scoring option: If you'd like to run a flat scorer that does not apply
    the scaled sigmoid weighting, comment out the two scaledSigmoid() lines
    below, and uncomment the replacement lines to calculate thisTP and thisFP.

    @return  (float)    Score at each timestamp of the datafile.
    """

    # Scoring section (i) handles TP and FN, (ii) handles FP, and TN are 0.
    # Input to the scoring function is var position: within a given window, the
    # position relative to the true anomaly.
    scores = pandas.DataFrame([0]*len(self.data), columns=["S(t)"])

    # (i) Calculate the score for each window. Each window will either have one
    # or more true positives or no predictions (i.e. a false negative). FNs
    # lead to a negative contribution, TPs a positive one.
    tpScore = 0
    fnScore = 0
    maxTP = scaledSigmoid(-1.0)
    for window in self.windows:
      tpIndex = window.getFirstTruePositive()

      if tpIndex == -1:
        # False negative; mark once for the whole window (at the start)
        thisFN = -self.costMatrix["fnWeight"]
        scores.iloc[window.indices[0]] = thisFN
        fnScore += thisFN
      else:
        # True positive
        position = -(window.indices[-1] - tpIndex + 1)/float(window.length)
        thisTP = scaledSigmoid(position)*self.costMatrix["tpWeight"] / maxTP
        # thisTP = self.costMatrix["tpWeight"]  # flat scoring
        scores.iloc[window.indices[0]] = thisTP
        tpScore += thisTP

    # Go through each false positive and score it. Each FP leads to a negative
    # contribution dependent on how far it is from the previous window.
    fpLabels = self.data[self.data["type"] == "fp"]
    fpScore = 0
    for i in fpLabels.index:
      windowId = self.getClosestPrecedingWindow(i)

      if windowId == -1:
        thisFP = -self.costMatrix["fpWeight"]
        scores.iloc[i] = thisFP
        fpScore += thisFP
      else:
        window = self.windows[windowId]
        position = abs(window.indices[-1] - i)/float(window.length-1)
        thisFP = scaledSigmoid(position)*self.costMatrix["fpWeight"]
        # thisFP = -self.costMatrix["fpWeight"]  # flat scoring
        scores.iloc[i] = thisFP
        fpScore += thisFP

    self.score = tpScore + fpScore + fnScore

    return (scores, self.score)


  def getClosestPrecedingWindow(self, index):
    """Given a record index, find the closest preceding window.

    This helps score false positives.

    @param  index   (int)   Index of a record.

    @return         (int)   Window id for the last window preceding the given
                            index.
    """
    minDistance = float("inf")
    windowId = -1
    for window in self.windows:
      if window.indices[-1] < index:
        dist = index - window.indices[-1]
        if dist < minDistance:
          minDistance = dist
          windowId = window.id

    return windowId


def sigmoid(x):
  """Standard sigmoid function."""
  return 1 / (1 + math.exp(-x))


def scaledSigmoid(relativePositionInWindow):
  """Return a scaled sigmoid function given a relative position within a
  labeled window.  The function is computed as follows:

  A relative position of -1.0 is the far left edge of the anomaly window and
  corresponds to S = 2*sigmoid(5) - 1.0 = 0.98661.  This is the earliest to be
  counted as a true positive.

  A relative position of -0.5 is halfway into the anomaly window and
  corresponds to S = 2*sigmoid(0.5*5) - 1.0 = 0.84828.

  A relative position of 0.0 consists of the right edge of the window and
  corresponds to S = 2*sigmoid(0) - 1 = 0.0.

  Relative positions > 0 correspond to false positives increasingly far away
  from the right edge of the window. A relative position of 1.0 is past the
  right  edge of the  window and corresponds to a score of 2*sigmoid(-5) - 1.0 =
  -0.98661.

  @param  relativePositionInWindow (float)  A relative position
                                            within a window calculated per the
                                            rules above.

  @return (float)
  """
  if relativePositionInWindow > 3.0:
    # FP well behind window
    return -1.0
  else:
    return 2*sigmoid(-5*relativePositionInWindow) - 1.0


def scoreCorpus(threshold, args):
  """Scores the corpus given a detector's results and a user profile.

  Scores the corpus in parallel.

  @param threshold  (float)   Threshold value to convert an anomaly score value
                              to a detection.

  @param args       (tuple)   Contains:

    pool                (multiprocessing.Pool)  Pool of processes to perform
                                                tasks in parallel.
    detectorName        (string)                Name of detector.

    profileName         (string)                Name of scoring profile.

    costMatrix          (dict)                  Cost matrix to weight the
                                                true positives, false negatives,
                                                and false positives during
                                                scoring.
    resultsDetectorDir  (string)                Directory for the results CSVs.

    resultsCorpus       (nab.Corpus)            Corpus object that holds the per
                                                record anomaly scores for a
                                                given detector.
    corpusLabel         (nab.CorpusLabel)       Ground truth anomaly labels for
                                                the NAB corpus.
    probationaryPercent (float)                 Percent of each data file not
                                                to be considered during scoring.
  """
  (pool,
   detectorName,
   profileName,
   costMatrix,
   resultsDetectorDir,
   resultsCorpus,
   corpusLabel,
   probationaryPercent,
   scoreFlag) = args

  args = []
  for relativePath, dataSet in resultsCorpus.dataFiles.iteritems():
    if "_scores.csv" in relativePath:
      continue

    # relativePath: raw dataset file,
    # e.g. 'artificialNoAnomaly/art_noisy.csv'
    relativePath = convertResultsPathToDataPath( \
      os.path.join(detectorName, relativePath))

    # outputPath: dataset results file,
    # e.g. 'results/detector/artificialNoAnomaly/detector_art_noisy.csv'
    relativeDir, fileName = os.path.split(relativePath)
    fileName =  detectorName + "_" + fileName
    outputPath = os.path.join(resultsDetectorDir, relativeDir, fileName)

    windows = corpusLabel.windows[relativePath]
    labels = corpusLabel.labels[relativePath]

    probationaryPeriod = getProbationPeriod(
      probationaryPercent, labels.shape[0])

    predicted = convertAnomalyScoresToDetections(
      dataSet.data["anomaly_score"], threshold)

    args.append((
      detectorName,
      profileName,
      relativePath,
      outputPath,
      threshold,
      predicted,
      windows,
      labels,
      costMatrix,
      probationaryPeriod,
      scoreFlag))

  # Using `map_async` instead of `map` so interrupts are properly handled.
  # See: http://stackoverflow.com/a/1408476
  results = pool.map_async(scoreDataSet, args).get(99999999)

  # Total the 6 scoring metrics for all data files
  totals = [None]*3 + [0]*10
  for row in results:
    for i in xrange(10):
      totals[i+3] += row[i+4]

  results.append(["Totals"] + totals)

  resultsDF = pandas.DataFrame(data=results,
                               columns=("Detector", "Profile", "File",
                                        "Threshold", "Score", "TP", "TN",
                                        "FP", "FN", "Total_Count","recall_numerator", "rar_count", "precision_numerator", "par_count"))

  return resultsDF


def scoreDataSet(args):
  """Function called to score each dataset in the corpus.

  @param args   (tuple)  Arguments to get the detection score for a dataset.

  @return       (tuple)  Contains:
    detectorName  (string)  Name of detector used to get anomaly scores.

    profileName   (string)  Name of profile used to weight each detection type.
                            (tp, tn, fp, fn)

    relativePath  (string)  Path of dataset scored.

    threshold     (float)   Threshold used to convert anomaly scores to
                            detections.

    score         (float)   The score of the dataset.

    counts, tp    (int)     The number of true positive records.

    counts, tn    (int)     The number of true negative records.

    counts, fp    (int)     The number of false positive records.

    counts, fn    (int)     The number of false negative records.

    total count   (int)     The total number of records.
  """
  (detectorName,
   profileName,
   relativePath,
   outputPath,
   threshold,
   predicted,
   windows,
   labels,
   costMatrix,
   probationaryPeriod,
   scoreFlag) = args

  scorer = Scorer(
    timestamps=labels["timestamp"],
    predictions=predicted,
    labels=labels["label"],
    windowLimits=windows,
    costMatrix=costMatrix,
    probationaryPeriod=probationaryPeriod)

  (scores,_) = scorer.getScore()

  pars= []
  rars= []
  recallNumerator = 0.0
  precisionNumerator = 0.0
 

  if scoreFlag:
    # Append scoring function values to the respective results file
    df_csv = pandas.read_csv(outputPath, header=0, parse_dates=[0])
    df_csv["S(t)_%s" % profileName] = scores

    #TJ's code
    df_csv["prediction_%s" % profileName] = pandas.Series([-1], index=df_csv.index)
    df_csv["prediction_%s" % profileName] = predicted

    predicted_lists = predicted.tolist()
    pars = compute_anomaly_windows(predicted_lists)

    label_lists = labels["label"].tolist()
    rars = compute_anomaly_windows(label_lists)

    recallNumerator, _ = compute_recall(pars, rars, 0.8, flat)
    precisionNumerator, _ = compute_precision(pars, rars, flat)
    df_csv.to_csv(outputPath, index=False)

  counts = scorer.counts

  return (detectorName, profileName, relativePath, threshold, scorer.score,
    counts["tp"], counts["tn"], counts["fp"], counts["fn"], scorer.length, recallNumerator, len(rars), precisionNumerator, len(pars))
