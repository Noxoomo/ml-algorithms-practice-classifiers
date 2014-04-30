package RandomForest;

import java.util.LinkedList;
import java.util.concurrent.RecursiveTask;

/**
 * User: Vasily
 * Date: 27.04.14
 * Time: 17:40
 */
public class ForestGrower extends RecursiveTask<LinkedList<Node>> {
    private final int treesCount;
    private int m;
    private DataFrame df;
    private int workersCount = 4;
    private boolean sequential = false;

    public ForestGrower(DataFrame df, int m,int treesCount) {
        this.m = m;
        this.df = df;
        this.treesCount = treesCount;

    }


    public ForestGrower(DataFrame df, int m,int treesCount,boolean sequential) {
        this(df,m,treesCount);
        this.sequential = sequential;
    }

    @Override
    protected LinkedList<Node> compute() {
        LinkedList<Node> forest = new LinkedList<>();
        if (sequential || treesCount < workersCount) {
            TreeGrower grower = new TreeGrower(df,m);
            for (int i=0;i<treesCount;++i)
                forest.add(grower.growTree());
            return forest;
        }

        LinkedList<ForestGrower> growers = new LinkedList<>();
        int treesPerThread  = (int) Math.ceil(treesCount / workersCount);      // it's random forest, +1 tree, -1 tree — not so important


        for (int i = 0; i < workersCount; ++ i) {
            ForestGrower grower = new ForestGrower(df,m,treesPerThread,true);
            growers.add(grower);
        }
        invokeAll(growers);

        for (ForestGrower grower:growers) {
            forest.addAll(grower.join());
        }
//        int toFirst = treesCount / workersCount;
//        int toSecond = treesCount - toFirst;
//        RandomForest.ForestGrower first = new RandomForest.ForestGrower(df,m,toFirst);
//        RandomForest.ForestGrower second = new RandomForest.ForestGrower(df,m, toSecond);
//        first.fork();
//        second.fork();
//        forest = first.join();
//        forest.addAll(second.join());
        return forest;
    }
}
