package SVM;

import java.util.Random;

/**
 * User: Vasily
 * Date: 30.11.13
 * Time: 9:42
 */
public class SVM extends SVMAbstract {
    //svm params
    double C = 1;
    //fit svm with this tolerance
    double tolerance = 1e-2;
    double eps = 1e-2;
    // first coordinate — num of observation vector
    // => learn[1][1] - first 'user', first feature,   learn[1][2] - first user, second feature
    //learn data
    private double learn[][];
    private double alpha[];
    private double beta;
    private double category[];
    private int maxTries = 100;
    private int steps = 1;
    private Random rand = new Random();
    //cache
//    private HashMap<Integer, Double> error = new HashMap<Integer, Double>();
    private double errorCache[];
    private int errorActive[];
    private boolean falseArray[];
    private double kernelCache[][];
    private boolean violate = false;

    public SVM(double learn[][], double category[], double C) {
        this(learn, category);
        this.C = C;
    }

    public SVM(double learn[][], double category[]) {
        this.category = category;
        this.learn = learn;
        this.alpha = new double[category.length];
        this.errorCache = new double[category.length];
        this.errorActive = new int[category.length];
        this.falseArray = new boolean[category.length];
        //  error = new double[category.length];
    }

    private void cache() {
        kernelCache = new double[alpha.length][alpha.length];
        for (int i = 0; i < alpha.length; ++i)
            for (int j = i; j < alpha.length; ++j) {
                kernelCache[i][j] = kernel(learn[i], learn[j]);
                kernelCache[j][i] = kernelCache[i][j];
            }
    }

    private boolean step(int maxTries) {
        violate = false;
        boolean updated = false;
        int start = rand.nextInt(alpha.length - 1);
        for (int i = start + 1; i != start; i = (i + 1) % alpha.length) {
            double err;
            if (errorActive[i]!=steps) {
                err = f(i) - category[i];
                errorCache[i] = err;
                errorActive[i] = steps;
            } else {
                err = errorCache[i];
            }
            double r = category[i] * err;
            if ((r > tolerance && alpha[i] > 0) || (r < -tolerance && alpha[i] < C)) {
                if (optimize(i, err, maxTries))
                    updated = true;
                violate = true;
            }
        }
        double err = f(start) - category[start];
        double r = category[start] * err;
        if ((r > tolerance && alpha[start] > 0) || (r < -tolerance && alpha[start] < C)) {
            violate = true;
            updated = optimize(start, err, maxTries);
        }
        return updated;
    }

    void learn() {
        cache();
        boolean updated = true;
        while (updated) {
            updated = step(maxTries);
        }
        int supVectCount = 0;
        for (double anAlpha : alpha) {
            if (anAlpha > 0)
                supVectCount++;
        }
        System.err.print("Steps: ");
        System.err.println(steps);
        System.err.print("Support vector's: ");
        System.err.println(supVectCount);
    }



    private boolean optimize(int i, double err1, int optimizeTries) {
        for (int k = 0; k < optimizeTries; ++k) {
            int j = rand.nextInt(alpha.length);
            if (step(i, j, err1)) return true;
        }
        return false;
    }

    double f(int i) {
        double result = 0;
        for (int k = 0; k < alpha.length; ++k) {
            if (alpha[k] == 0) continue;
            result += alpha[k] * category[k] * (kernel(k, i));
        }
        return result - beta;
    }

    double f(double observation[]) {
        double result = 0;
        for (int i = 0; i < alpha.length; ++i) {
            if (alpha[i] == 0) continue;
            result += alpha[i] * category[i] * (kernel(learn[i], observation));
        }
        return result - beta;
    }

    double predict(double observation[]) {
        return sign(f(observation));
    }

    @Override
    double[] predict(double observation[][]) {
        double result[] = new double[observation.length];
        for (int i = 0; i < observation.length; ++i) {
            result[i] = predict(observation[i]);
        }
        return result;
    }

    int sign(double d) {
        if (d > 0) return 1;
        if (d < 0) return -1;
        return 0;
    }

    @Override
    double kernel(double u[], double v[]) {
        if (u.length != v.length) {
            throw new RuntimeException();
        }
        double result = 0;
        for (int i = 0; i < u.length; ++i) {
            result += u[i] * v[i];
        }
        return result;
    }

    private double kernel(int i, int j) {
        return kernelCache[i][j];
    }

    private double calcBeta(double aiNew, double ajNew, int i, int j, double err1, double err2) {
        double b1 = beta + err1 + category[i] * (aiNew - alpha[i]) * kernel(i, i) + category[j] * (ajNew - alpha[j]) * kernel(i, j);
        double b2 = beta + err2 + category[i] * (aiNew - alpha[i]) * kernel(i, j) + category[j] * (ajNew - alpha[j]) * kernel(j, j);
        return (b1 + b2) / 2;
    }

    private boolean step(int i, int j, double err1) {
        if (i == j) return false;
        double err2;
        if (errorActive[j]!=steps) {
            err2 = f(j) - category[j];
            errorActive[j] = steps;
            errorCache[j] = err2;
        } else {
            err2 = errorCache[j];
        }
        double s = category[i] * category[j];
        double L;
        double H;
        if (category[i] != category[j]) {
            L = max(0, alpha[j] - alpha[i]);
            H = min(C, C + alpha[j] - alpha[i]);
        } else {
            L = max(0, alpha[i] + alpha[j] - C);
            H = min(C, alpha[i] + alpha[j]);
        }
        if (abs(H - L) < eps) return false;
        double eta = 2 * kernel(i, j) - kernel(i, i) - kernel(j, j);
        if (eta >= 0) {
            System.err.println("eta>=0");
            return false;
        }
        double ajNew = alpha[j] + category[j] * (err2 - err1) / eta;
        if (ajNew < L)
            ajNew = L;
        else if (ajNew > H)
            ajNew = H;
        //ai new
        if (abs(ajNew) < eps) {
            ajNew = 0;
        }
        if (ajNew > C - eps) {
            ajNew = C;
        }
        double aiNew = alpha[i] - s * (ajNew - alpha[j]);
        if (abs(alpha[j] - ajNew) < eps) return false;
        if (aiNew < eps ) aiNew = 0;
        if (aiNew > C - eps) aiNew = C;
        //new beta
        double newBeta = calcBeta(aiNew, ajNew, i, j, err1, err2);
        double betaOld = beta;
        beta = newBeta;
       // System.arraycopy(falseArray, 0, errorActive, 0, falseArray.length);


        alpha[i] = aiNew;
        alpha[j] = ajNew;
        steps++;
        return true;
    }

    private double abs(double a) {
        if (a >= 0) return a;
        return -a;
    }

    private double min(double a, double b) {
        if (a < b) return a;
        return b;
    }

    private double max(double a, double b) {
        if (a > b) return a;
        return b;
    }


}
