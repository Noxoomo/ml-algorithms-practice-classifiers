package NeuralNet;

import java.util.concurrent.RecursiveTask;

/**
 * User: Vasily
 * Date: 30.04.14
 * Time: 18:53
 */
public class NetworkChooser extends RecursiveTask<NeuralNet> {
    private int[] topology;
    private double[][] learn;
    private int[] labels;
    private double[][] validate;
    private int[] validateLabels;
    private double validateSplit = 0.75;
    private int count = 0;

    public NetworkChooser(int[] topology, double[][] learn, int[] labels, double[][] validate, int[] validateLabels, int count) {
        this.topology = topology;
        this.labels = labels;
        this.learn = learn;
        this.validate = validate;
        this.validateLabels = validateLabels;
        this.count = count;
    }

    @Override
    protected NeuralNet compute() {
        if (count <= 4) {
            NeuralNet bestNetwork = new NeuralNet(topology,learn, labels,validateSplit);
            double bestKappa = bestNetwork.fitNetwork(validate, validateLabels);
            for (int i = 0; i < count; ++i) {
                NeuralNet net =  new NeuralNet(topology,learn, labels,validateSplit);
                double netKappa = net.fitNetwork(validate, validateLabels);
                if (netKappa > bestKappa) {
                    bestKappa = netKappa;
                    bestNetwork = net;
                }
            }
//            System.err.println(count);
            return bestNetwork;

        }
        NetworkChooser left = new NetworkChooser(topology, learn, labels, validate, validateLabels, count / 2);
        NetworkChooser right = new NetworkChooser(topology, learn, labels, validate, validateLabels, count - count / 2);
        invokeAll(left,right);
        NeuralNet leftBest = left.join();
        NeuralNet rightBest = right.join();

//        int chunkSize  = (int)ceil(1.0*count / threads);
//        LinkedList<NetworkChooser> choosers = new LinkedList<>();
//        for (int i =0;i<threads-1;++i) {
//            NetworkChooser chooser = new NetworkChooser(topology,learn,labels,validate,validateLabels,1,chunkSize);
//            chooser.fork();
//            choosers.add(chooser);
//        }
//        choosers.add(new NetworkChooser(topology, learn, labels, validate, validateLabels, 1, count - (threads - 1) * chunkSize));
//        choosers.getLast().fork();
//
//        NeuralNet best = choosers.pop().join();
//        for (NetworkChooser chooser : choosers) {
//            NeuralNet candidate = chooser.join();
//            if (candidate.testKappa > best.testKappa) {
//                best = candidate;
//            }
//        }
        return leftBest.testKappa > rightBest.testKappa ? leftBest : rightBest;
    }


}
