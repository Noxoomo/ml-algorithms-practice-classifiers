import java.util.LinkedList;
import java.util.Random;

/**
 * User: Vasily
 * Date: 25.04.14
 * Time: 21:34
 */
public class RandomForestClassifier {
    private DataFrame df;
    //    private int[] labels;
    private LinkedList<Node> trees; //root of trees
    private int labelsCount;
    int maxDepth = -1;
    private int m = -1;
    private final int minLeafSize = 3;
    private Random random = new Random();


    public RandomForestClassifier(double[][] observations, int[] labels, int m, int maxDepth, int labelsCount) throws Exception {
        this.df = new DataFrame(observations, labels);
        this.m = m;
        this.maxDepth = maxDepth;
        this.labelsCount = labelsCount;
    }


    public void addTrees(int forestsCount) {
        for (int i = 0; i < forestsCount; ++i) {
            trees.add(this.growTree());
        }
    }

    private Node growTree() {
        int[] index = new int[df.featuresCount];
        for (int i = 0; i < index.length; ++i) {
            index[i] = random.nextInt(df.featuresCount);
        }
        return growTree(index);
    }

    private Node growTree(int[] index) {
        if (index.length < minLeafSize) {
            Node root = new Node(df.predict(index));
            return root;
        }

        int featuresToProceed = m == -1 ? df.featuresCount : m;
        int[] featuresSet = chooseFeatures(featuresToProceed, df.featuresCount);

        LinkedList<SplitPoint> splits = new LinkedList<>();
        for (int feature : featuresSet) {
            splits.add(df.findSplit(feature, index, minLeafSize));
        }
        SplitPoint best = splits.pop();
        for (SplitPoint candidate : splits) {
            if (candidate.gini > best.gini) {
                best = candidate;
            }
        }

        int[] leftIndex = new int[best.splitIndex];
        int[] rightIndex = new int[index.length - best.splitIndex];
        System.arraycopy(index, 0, leftIndex, 0, best.splitIndex);
        System.arraycopy(index, 0, rightIndex, 0, index.length - best.splitIndex);
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


    public int[] predict(int[][] testObservations) {
        int[] result = new int[testObservations.length];
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

    public int predict(int[] observation) {
        if (!isLeaf) {
            if (observation[feature] >= threshold) {
                return right.predict(observation);
            } else {
                return left.predict(observation);
            }

        } else {
            return value;
        }
    }
}

