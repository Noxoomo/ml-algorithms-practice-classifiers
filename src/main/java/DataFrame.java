import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

/**
 * User: Vasily
 * Date: 26.04.14
 * Time: 23:59
 */
public class DataFrame {
    private double[][] data;
    private int[] index;
    private Random rand = new Random();
    public final int featuresCount;
    public final int observationsCount;
    private int[] labels;
    private int labelsCount;


    public DataFrame(double[][] data, int[] labels) {
        this.data = data;
        this.index = new int[data[0].length];
        this.featuresCount = data.length;
        this.observationsCount = this.index.length;
        for (int i = 0; i < this.index.length; ++i) {
            this.index[i] = i;
        }
        this.labels = labels;
        Set<Integer> labelsSet = new TreeSet<>();
        for (int label : labels) {
            labelsSet.add(label);
        }
        this.labelsCount = labelsSet.size();

    }


    //return argsort of i'th column
    public int[] argsort(int i) {
        int[] order = new int[index.length];
        System.arraycopy(index, 0, order, 0, index.length);
        qsort(i, order, 0, order.length - 1);
        return order;
    }

    public int[] argsort(int i, int[] index) {
        int[] order = new int[index.length];
        System.arraycopy(index, 0, order, 0, index.length);
        qsort(i, order, 0, order.length - 1);
        return order;
    }

    public double get(int i, int j, int[] index) {
        return data[i][index[j]];
    }

    private void qsort(int i, int[] order, int left, int right) {
        int l = left;
        int r = right;
        if (r - l <= 0) {
            return;
        }
        if (r - l == 1) {
            if (data[i][order[l]] > data[i][order[r]]) {
                swap(l, r, order);
            }
            return;
        }

        int mid = l + rand.nextInt(r - l);
        double pivot = data[i][order[mid]];

        while (l < r) {
            while (data[i][order[l]] < pivot) {
                l++;
            }
            while (data[i][order[r]] > pivot) {
                r--;
            }
            if (l <= r) {
                swap(l, r, order);
                l++;
                r--;
            }
        }
        if (r > left) {
            qsort(i, order, left, r);
        }
        if (l < right) {
            qsort(i, order, l, right);
        }
    }

    private void swap(int l, int r, int[] order) {
        int tmp = order[l];
        order[l] = order[r];
        order[r] = tmp;
    }


    public SplitPoint findSplit(int feature, int[] index, int minLeafSize) {
        //initialize
        int[] order = argsort(feature, index);
        int currentSplit = minLeafSize;
        long leftSumSquares = 0;
        long rightSumSquares = 0;

        int[] leftLabelsCounts = new int[labelsCount];
        int[] rightLabelsCounts = new int[labelsCount];

        for (int ind = 0; ind < currentSplit; ++ind) {
            leftLabelsCounts[labels[order[ind]]]++;
        }
        for (int ind = currentSplit; ind < order.length; ++ind) {
            rightLabelsCounts[labels[order[ind]]]++;
        }


        for (int labelCount : rightLabelsCounts) {
            rightSumSquares += labelCount * labelCount;
        }
        for (int labelCount : leftLabelsCounts) {
            leftSumSquares += labelCount * labelCount;
        }
        SplitPoint best = new SplitPoint();
        best.feature = feature;
        best.threshold = 0.5 * (data[feature][order[currentSplit]] - data[feature][order[currentSplit - 1]]);
        best.gini = 1.0 * leftSumSquares / currentSplit + 1.0 * rightSumSquares / (order.length - currentSplit);
        best.splitIndex = currentSplit;

        while (currentSplit < index.length - minLeafSize) {
            int label = labels[order[currentSplit]];
            leftSumSquares += 2 * leftLabelsCounts[label] + 1;
            rightSumSquares -= 2 * rightLabelsCounts[label] - 1;

            leftLabelsCounts[label] += 1;
            rightLabelsCounts[label] -= 1;
            currentSplit++;

            double newGini = 1.0 * leftSumSquares / currentSplit + 1.0 * rightSumSquares / (order.length - currentSplit);
            if (newGini > best.gini) {
                best.gini = newGini;
                best.threshold = 0.5 * (data[feature][order[currentSplit]] - data[feature][order[currentSplit - 1]]);
                best.splitIndex = currentSplit;
            }
        }
        return best;
    }

    public int predict(int[] index) {
        int[] votes = new int[labelsCount];
        for (int ind : index) {
            votes[labels[index[ind]]]++;
        }
        int best = votes[0];
        int bestLabel = 0;
        for (int i = 1; i < votes.length; ++i) {
            if (votes[i] > best) {
                best = votes[i];
                bestLabel = i;
            }
        }
        return bestLabel;
    }
}
