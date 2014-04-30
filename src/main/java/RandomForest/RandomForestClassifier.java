package RandomForest;

import java.util.LinkedList;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

/**
 * User: Vasily
 * Date: 25.04.14
 * Time: 21:34
 */
public class RandomForestClassifier {
    private DataFrame df;
    //    private int[] labels;
    private LinkedList<Node> trees = new LinkedList<>(); //root of trees
    private int labelsCount;
    //    int maxDepth = -1;
    private int m = -1;
    private final int minLeafSize = 1;
    private Random random = new Random();
    private int maxThreads = 4;
    private ForkJoinPool pool = new ForkJoinPool(maxThreads);


    public RandomForestClassifier(double[][] observations, int[] labels, int m, int labelsCount) {
        this.df = new DataFrame(observations, labels);
        this.m = m;
//        this.maxDepth = maxDepth;
        this.labelsCount = labelsCount;
    }


    public void addTrees(int treesCount) {
        ForestGrower grower = new ForestGrower(df,m,treesCount);
        trees.addAll(pool.invoke(grower));
//        for (int i = 0; i < forestsCount; ++i) {
//            trees.add(this.growTree());
//        }
    }



    public int[] predict(double[][] testObservations) {
        int[] result = new int[testObservations.length];
        for (int i=0;i< result.length;++i)
            result[i] = -1;
        for (int i = 0; i < testObservations.length; ++i) {
            int[] votes = new int[labelsCount];
            for (Node tree : trees) {
                votes[tree.predict(testObservations[i])]++;
            }
            int best = 0;
            int bestVotes = votes[0];
            for (int j = 1; j < votes.length; ++j) {
                if (bestVotes < votes[j]) {
                    bestVotes = votes[j];
                    best = j;
                }
            }
            result[i] = best;
        }
        return result;
    }
}


class SplitPoint {
    public int feature;
    public double threshold;
    public int splitIndex;
    public double gini;
    public int[] order;
}

class Node {
    public int feature;
    public double threshold;
    public Node left = null;
    public Node right = null;
    private boolean isLeaf = false;
    private int value;

    Node(int feature, double threshold) {
        this.feature = feature;
        this.threshold = threshold;
    }

    Node(int value) {
        isLeaf = true;
        this.value = value;
    }

    public int predict(double[] observation) {
        if (!isLeaf) {
            if (observation[feature] > threshold) {
                return right.predict(observation);
            } else {
                return left.predict(observation);
            }

        } else {
            return value;
        }
    }
}

