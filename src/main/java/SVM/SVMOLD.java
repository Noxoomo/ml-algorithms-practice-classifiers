package SVM;

import java.util.HashSet;
import java.util.Set;

/**
 * User: Vasily
 * Date: 30.11.13
 * Time: 9:42
 */
public class SVMOLD {
    // first coordinate — num of observation vector
    // => learn[1][1] - first 'user', first feature,   learn[1][2] - first user, second feature
    //learn data
    private double learn[][];
    private double alpha[];
    private double beta;
    double category[];
    int steps = 0;

    //svm params
    double C  = 1;
    //fit svm with this tolerance
    double tolerance = 1e-2;
    double eps = 1e-4;

    //cache
    private double error[];
    private double kernelCache[][];
    private Set<Integer> activeSet;


    public SVMOLD(double learn[][], double category[], double C) {
        this(learn,category);
        this.C = C;
    }

    public SVMOLD(double learn[][], double category[]) {
        this.category = category;
        this.learn = learn;
        this.alpha = new double[category.length];
        kernelCache = new double[alpha.length][alpha.length];
        for (int i = 0; i < alpha.length; ++i)
            for (int j = i; j < alpha.length; ++j) {
                kernelCache[i][j] = kernel(learn[i], learn[j]);
                kernelCache[j][i] = kernelCache[i][j];
            }
        activeSet = new HashSet<Integer>();
        error = new double[category.length];
        learn();
    }
    private void  learn() {
        boolean updated = true;
        while (updated) {
            updated = false;
            for (int i=0; i < alpha.length;++i) {
                if (visit(i)) {
                    updated = optimize(i);
                    break;
                }
            }
        }
    }

    private boolean visit(int i) {
        double err;
        if (alpha[i]  > 0 && alpha[i] < C) {
            err = error[i];
        }  else {
            err = f(learn[i]) - category[i];
            error[i] = err;
        }
        double r = category[i] * err;
        if (r > tolerance && alpha[i] > 0) {
            return true;
        }
        if (r < -tolerance && alpha[i] < C) {
            return true;
        }
        return false;
    }


    private boolean firstHeuristic(int i) {
        boolean set = false;
        double best  = 0;
        int bestNum = 0;
        for (int j: activeSet) {
            if (j == i) continue;
            if (!set || abs(error[i] - error[j]) > best) {
                set = true;
                best = error[j];
                bestNum = j;
            }
        }
        return step(i,bestNum);
    }
    private boolean secondHeuristic(int i) {
        for (int j: activeSet) {
            if (i==j) continue;
            if (step(i,j)) return true;
        }
        return false;
    }
    private  boolean tryAll(int i) {
        for (int j = 0; j < alpha.length;++j) {
            if (!activeSet.contains(j)) {
                if (step(i,j)) return true;
            }
        }
        return false;
    }
    private boolean optimize(int i) {
        if (activeSet.size() > 0 && firstHeuristic(i)) return true;
        else if (activeSet.size() > 0 && secondHeuristic(i)) return true;
        else return tryAll(i);
    }


    double f(double observation[]) {
        double result = 0;
        for (int i=0;i<alpha.length;++i) {
            if (alpha[i] ==0) continue;
            result +=  alpha[i]*category[i]*(kernel(learn[i],observation));
        }
        return result-beta;
    }

    double predict(double observation[]) {
        return sign(f(observation));
    }
    double[] predict(double observation[][]) {
        double result[] = new double[observation.length];
        for (int i=0;i<observation.length;++i) {
            result[i] = predict(observation[i]);
        }
        return result;
    }




    int sign(double d) {
        if (d > 0) return 1;
        if (d < 0) return -1;
        return 0;
    }

    private double kernel(double u[], double v[]) {
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



    private boolean step(int i, int j) {
        if (i == j) return false;
        if (alpha[j]  == 0 || alpha[j] == C)
            error[j] = f(learn[j]) - category[j];
        double L;
        double H;
        if (category[i] != category[j]) {
            L = max(0,alpha[j] - alpha[i]);
            H = min(C,C+alpha[j]-alpha[i]);
        } else {
            L = max(0,alpha[i] + alpha[j] -C);
            H = min(C,alpha[i]+alpha[j]);
        }
        if (abs(H-L) < eps) return false;
        double eta = 2*kernel(i,j) - kernel(i,i) - kernel(j,j);
        double ajNew;
        //cacl new alpha[j]
        if (eta < 0) {
            ajNew = alpha[j] - category[j] * (error[i] - error[j]) / eta;
            ajNew = clipped(ajNew,H,L);
        } else {
            double c1 = eta / 2;
            double c2 = category[j] * (error[i]-error[j]) - eta*alpha[j];
            double targetL = c1*L*L+c2*L;
            double targetH = c1*H*H+c2*H;
            if (targetL - targetH > eps) {
                ajNew = L;
            }  else if (targetH - targetL > eps) {
                ajNew = H;
            } else ajNew = alpha[j];
        }
        if (ajNew < eps) {
            ajNew = 0;
        }
        if (ajNew > C - eps) {
            ajNew = C;
        }
        if (abs(alpha[j] - ajNew) < eps) return false;
        //calc new alpha[i]
        double aiNew = alpha[i] + category[i]*category[j] * (alpha[j] - ajNew);
        double betaOld = beta;

        if (aiNew > 0 && aiNew < C) {
            beta =  error[i]  + category[i] * (aiNew - alpha[i]) * kernel(i,i)
                    + category[j]*(ajNew-alpha[j])*kernel(i,j) + beta;

        } else if (ajNew > 0 && ajNew < C) {
            beta = error[j] + category[i]*(aiNew - alpha[i]) * kernel(i,j) +
                    category[j] * (ajNew - alpha[j]) * kernel(j,j) + beta;

        } else {
        double b1 = error[i]  + category[i] * (aiNew - alpha[i]) * kernel(i,i)
                + category[j]*(ajNew-alpha[j])*kernel(i,j) + beta;
        double b2 = error[j] + category[i]*(aiNew - alpha[i]) * kernel(i,j) +
                category[j] * (ajNew - alpha[j]) * kernel(j,j) + beta;
        beta = (b1 + b2) / 2;
        }

        for (int k=0; k < error.length;++k) {
            if (alpha[k]!=0 && alpha[k]!=C)  {
                error[k] = error[k] + category[i]*(aiNew - alpha[i]) * kernel(i,k) +
                    category[j] * (ajNew - alpha[j]) * kernel(j,k) - beta + betaOld;

            }
        }
        error[i] = 0;
        error[j] = 0;

        alpha[i] = aiNew;
        alpha[j] = ajNew;
//        for (int k=0;k<error.length;++k) {
//            double err = f(learn[k]) - category[k];
//            if (k !=i && k!=j && alpha[k]> 0 && alpha[k]<C && abs(err - error[k]) > 1e-4) {
//                System.out.println("aaa");
//                break;
//            }
//        }
        if (alpha[i] == 0 || alpha[i] == C) activeSet.remove(i);
        if (alpha[j] == 0 || alpha[j] == C) activeSet.remove(j);
        if (alpha[i] > 0 && alpha[i] < C) activeSet.add(i);
        if (alpha[j] > 0 && alpha[j] < C) activeSet.add(j);
        steps++;
        return true;
    }

    private double abs(double a) {
        if (a >= 0) return a;
        return  -a;
    }
    private double min(double a,double b) {
        if (a < b) return a;
        return b;
    }
    private double max(double a, double b) {
        if (a > b) return a;
        return b;
    }
    private double clipped(double a, double H, double L) {
        if (a > H) return H;
        else if (a > L) return  a;
        return L;
    }
}
