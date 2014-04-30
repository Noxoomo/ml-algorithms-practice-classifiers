package RandomForest;

import java.util.LinkedList;
import java.util.Random;

/**
 * User: Vasily
 * Date: 27.04.14
 * Time: 17:11
 */

public class TreeGrower {
    private DataFrame df;
    private int m;
    private Random random = new Random();
    private int minLeafSize = 1;

    public TreeGrower(DataFrame df, int m) {
        this.df = df;
        this.m = m;
    }

    public TreeGrower(DataFrame df, int m, int minLeafSize) {
        this(df, m);
        this.minLeafSize = minLeafSize;
    }

    public Node growTree() {
        int[] index = new int[df.observationsCount];
        for (int i = 0; i < index.length; ++i) {
            index[i] = random.nextInt(df.observationsCount);
        }
        return growTree(index);
    }

    public Node growTree(int[] index) {
        if (index.length <= minLeafSize) {
            return new Node(df.predict(index));
        }
        int featuresToProceed = m == -1 ? df.featuresCount : m;
        int[] featuresSet = chooseFeatures(featuresToProceed, df.featuresCount);
        LinkedList<SplitPoint> splits = new LinkedList<>();
        for (int feature : featuresSet) {
            if (df.canSplit(feature, index)) {
                splits.add(df.findSplit(feature, index));
            }
        }
        if (splits.size() == 0) {
            return new Node(df.predict(index));
        }

        SplitPoint best = splits.pop();
        for (SplitPoint candidate : splits) {
            if (candidate.gini > best.gini) {
                best = candidate;
            }
        }
        int[] leftIndex = new int[best.splitIndex];
        int[] rightIndex = new int[index.length - best.splitIndex];
        System.arraycopy(best.order, 0, leftIndex, 0, best.splitIndex);
        System.arraycopy(best.order, best.splitIndex, rightIndex, 0, index.length - best.splitIndex);
        Node root = new Node(best.feature, best.threshold);
        root.left = growTree(leftIndex);
        root.right = growTree(rightIndex);
        return root;
    }


    private int[] chooseFeatures(int featuresToProceed, int featuresCount) {
        int[] features = new int[featuresToProceed];
        int[] featuresList = new int[featuresCount];
        for (int i = 0; i < featuresList.length; ++i)
            featuresList[i] = i;
        for (int i = featuresList.length - 1; i > 0; --i) {
            int ind = random.nextInt(i + 1);
            if (ind != i) {
                int tmp = featuresList[i];
                featuresList[i] = featuresList[ind];
                featuresList[ind] = tmp;
            }
        }
        System.arraycopy(featuresList, 0, features, 0, featuresToProceed);
        return features;
    }

}
